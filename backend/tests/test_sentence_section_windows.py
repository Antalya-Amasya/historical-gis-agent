from dataclasses import replace
import hashlib
import pytest
from backend.app.rag.ingestion.corpus_dry_run import document_chunks, sentence_windows, sentence_boundaries, SENTENCE_CHUNKER_VERSION
from backend.app.rag.ingestion.corpus_registry import CorpusDocument
from backend.app.rag.ingestion.document_sections import DocumentSection
from backend.app.rag.ingestion.production_lifecycle import CorpusIngestionConfig

DOC=CorpusDocument(document_id='doc',author='Author',work='Work',language='en',source_type='primary_source',filename='a.epub')
def section(text, index=0):
    return DocumentSection('doc','Author','Work',None,'1',None,None,'Book 1',text,'a.epub',0,'a.xhtml',index)
def chunks(text,**kwargs):
    return document_chunks([section(text)],DOC,chunker_version=SENTENCE_CHUNKER_VERSION,**kwargs)
def prose(n=80):
    return ' '.join(f'Marcus marched from Rome to Capua with the legion number {i}, and returned safely.' for i in range(n))

@pytest.mark.parametrize('text',['Short section.','Near target sentence. '*190,'   Unicode 凯撒 marched to Αθήνα.  ',''], ids=['short','near_target','unicode','empty'])
def test_short_near_target_unicode_and_empty_sections(text):
    out=chunks(text)
    assert ''.join(''.join(c.text.split()) for c in out)==''.join(text.split())
    assert all(c.metadata['chunker_version']==SENTENCE_CHUNKER_VERSION for c in out)
    if not text:assert out==[]

def test_long_multisentence_boundaries_do_not_start_or_end_mid_sentence():
    text=prose(200);out=chunks(text)
    assert len(out)>2
    for c in out:
        assert c.text.startswith('Marcus') and c.text.endswith('.')
        assert len(c.text)<=5000
    assert all(len(c.text)>=2600 for c in out[:-1])

def test_disjoint_spans_reconstruct_every_character_without_loss_or_duplication():
    text=prose(200)
    spans=list(sentence_windows(text))
    assert spans[0][0]==0 and spans[-1][1]==len(text)
    assert all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
    assert ''.join(text[a:b] for a,b,_ in spans)==text

def test_emitted_offsets_are_exact_after_whitespace_trimming():
    text='   '+prose(180)+'   '
    for c in chunks(text):
        assert text[c.metadata['start_offset']:c.metadata['end_offset']]==c.text

def test_two_sections_never_cross_and_have_distinct_identity():
    sections=[section(prose(80)),section(prose(80),1)]
    out=document_chunks(sections,DOC,chunker_version=SENTENCE_CHUNKER_VERSION)
    assert {c.metadata['section_index'] for c in out}=={0,1}
    assert len({c.id for c in out})==len(out)
    for c in out:
        source=sections[c.metadata['section_index']].text
        assert source[c.metadata['start_offset']:c.metadata['end_offset']]==c.text

def test_ids_preserve_existing_hash_form_and_are_deterministic():
    first=chunks(prose());second=chunks(prose())
    assert first==second
    for c in first:
        m=c.metadata
        expected=hashlib.sha256(f"doc:a.xhtml:0:{m['start_offset']}:{m['end_offset']}:{c.text}".encode()).hexdigest()[:32]
        assert c.id==expected

def test_changed_boundaries_generate_changed_ids_and_v1_default_is_preserved():
    text=prose(150)
    legacy=document_chunks([section(text)],DOC)
    new=chunks(text)
    assert legacy[0].metadata['chunker_version']=='section-window-v1'
    assert legacy[1].metadata['start_offset']==3400
    assert legacy[0].id!=new[0].id

@pytest.mark.parametrize('abbreviation',['B.C.','A.D.','Mr.','Dr.','cons.','cos.','ch.','chap.','IV.'])
def test_conservative_abbreviation_guard(abbreviation):
    text=f'The date or title is {abbreviation} Marcus marched. Next sentence.'
    boundaries=sentence_boundaries(text)
    assert text.index('Marcus') not in boundaries
    assert text.index('Next') in boundaries

def test_sentence_quotes_stay_with_the_sentence():
    text='He said, “Marcus marched.” Next he asked, "Where?" They replied.'
    positions=sentence_boundaries(text)
    assert text.index('Next') in positions and text.index('They') in positions
    assert ''.join(text[a:b] for a,b,_ in sentence_windows(text,25))==text

@pytest.mark.parametrize('text,expected',[('word '*2000,'raw'),('word '*600+'; '+'word '*1400,'secondary_punctuation'),('界'*10000,'raw')], ids=['long_words','secondary_punctuation','long_unicode'])
def test_oversized_sentence_fallback_is_bounded_and_explicit(text,expected):
    out=chunks(text)
    assert any(c.metadata.get('boundary_fallback')==expected for c in out)
    assert max(map(lambda c:len(c.text),out))<=5000
    assert ''.join(''.join(c.text.split()) for c in out)==''.join(text.split())

def test_unsupported_version_and_nonpositive_target_are_rejected():
    with pytest.raises(ValueError):document_chunks([section('Text.')],DOC,chunker_version='unknown')
    with pytest.raises(ValueError):list(sentence_windows('Text.',0))

def test_shadow_configuration_does_not_change_default_chunker():
    from pathlib import Path
    config=CorpusIngestionConfig('localhost',8002,'v2',Path('.'),'v2',('doc',))
    assert config.chunker_version=='section-window-v1'
    assert replace(config,chunker_version=SENTENCE_CHUNKER_VERSION,state_prefix='v3').state_path('journal').name=='journal_v3.json'

import ReactMarkdown from "react-markdown";

export function AnswerMarkdown({ answer }: { answer: string }) {
  return <ReactMarkdown>{answer}</ReactMarkdown>;
}

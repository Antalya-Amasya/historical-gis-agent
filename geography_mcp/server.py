from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from geography_mcp.service import GeographyService, GeoPoint
app=FastAPI(title="Historical Military GIS Geography MCP",version="0.3.0")
service=GeographyService()
class PlaceRequest(BaseModel): name:str; period:str|None=None
class DistanceRequest(BaseModel): point_a:GeoPoint; point_b:GeoPoint
class ProfileRequest(BaseModel): points:list[GeoPoint]
@app.get("/health")
def health(): return {"status":"ok","service":"geography-mcp"}
@app.post("/tools/resolve_ancient_place")
def resolve(request:PlaceRequest):
 return service.resolve_ancient_place_payload(request.name, request.period)
@app.post("/tools/calculate_distance")
def distance(request:DistanceRequest): return service.calculate_distance(request.point_a,request.point_b)
@app.post("/tools/get_elevation")
def elevation(point:GeoPoint): return service.get_elevation(point)
@app.post("/tools/get_elevation_profile")
def profile(request:ProfileRequest): return service.get_elevation_profile(request.points)

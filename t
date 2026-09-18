[1mdiff --git a/src/api/main.py b/src/api/main.py[m
[1mindex bae2950..527d3ae 100644[m
[1m--- a/src/api/main.py[m
[1m+++ b/src/api/main.py[m
[36m@@ -2,6 +2,7 @@[m
 import json[m
 [m
 from fastapi import FastAPI, HTTPException[m
[32m+[m[32mfrom fastapi.middleware.cors import CORSMiddleware[m
 [m
 from src.api.schemas import ([m
     AskRequest,[m
[36m@@ -18,6 +19,21 @@[m [mapp = FastAPI([m
 )[m
 [m
 [m
[32m+[m[32m# Allow the local Next.js frontend to call the backend.[m
[32m+[m[32m#[m
[32m+[m[32m# Local frontend:[m
[32m+[m[32m# http://localhost:3000[m
[32m+[m[32mapp.add_middleware([m
[32m+[m[32m    CORSMiddleware,[m
[32m+[m[32m    allow_origins=[[m
[32m+[m[32m        "http://localhost:3000",[m
[32m+[m[32m    ],[m
[32m+[m[32m    allow_credentials=True,[m
[32m+[m[32m    allow_methods=["*"],[m
[32m+[m[32m    allow_headers=["*"],[m
[32m+[m[32m)[m
[32m+[m
[32m+[m
 PROJECT_ROOT = Path(__file__).resolve().parents[2][m
 EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"[m
 [m
[36m@@ -156,6 +172,8 @@[m [mdef get_latest_evaluation():[m
     print("DEBUG: /eval/latest completed", flush=True)[m
 [m
     return result[m
[32m+[m
[32m+[m
 @app.post("/debug/ask-entry")[m
 def debug_ask_entry(request: AskRequest):[m
     """[m
[36m@@ -179,6 +197,8 @@[m [mdef debug_ask_entry(request: AskRequest):[m
         "k": request.k,[m
         "mode": request.mode,[m
     }[m
[32m+[m
[32m+[m
 @app.post("/debug/embedding")[m
 def debug_embedding():[m
     """[m
[36m@@ -206,4 +226,4 @@[m [mdef debug_embedding():[m
     return {[m
         "status": "ok",[m
         "dimensions": len(vector),[m
[31m-    }[m
[32m+[m[32m    }[m
\ No newline at end of file[m

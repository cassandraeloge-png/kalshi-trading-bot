"""Paper-only V3 forward BTC predictor.
Loads the exact trained V3 artifact and reproduces its 1-second feature pipeline
from public Binance.US BTCUSDT 1s klines. It never submits orders.
"""
import base64, math, os, pickle, tempfile, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

MODEL_URL=os.environ.get("V3_MODEL_URL","https://raw.githubusercontent.com/cassandraeloge-png/btc-15min/main/research/forward-predictor-v3-model.b64")
MODEL_BLOB_API=os.environ.get("V3_MODEL_BLOB_API","https://api.github.com/repos/cassandraeloge-png/btc-15min/git/blobs/24e04283a79c5a12ee3834f43b7fd209d44fcd95")
MODEL_PATH=Path(os.environ.get("V3_MODEL_PATH",str(Path(tempfile.gettempdir())/"forward-predictor-v3-model.b64")))
KLINES_URL="https://api.binance.com/api/v3/klines"
US_KLINES_URL="https://api.binance.us/api/v3/klines"
FEATURES=["move2","move3","move5","move10","move15","move30","move60","move120","accel5_15","accel15_30","imb5","imb15","imb30","imb60","imb_accel","vol5ratio","vol15ratio","trade5ratio","trade15ratio","range15","pos15","range60","pos60","range120","pos120","rv15","rv60","rv_ratio","body5","lowerwick5","round_sin","round_cos"]

class V3Predictor:
    def __init__(self):
        self.session=requests.Session(); self.bundle=None; self.last_fetch=0.0; self._reported_live=False
        self.last={"version":"v3","mode":"paper","status":"warming","source":"Binance.US BTCUSDT 1m → 1s proxy bars","note":"Live venue proxy; historical V3 was trained on Binance.com BTCUSDT.","next_15":None,"next_30":None,"next_60":None,"scalp_bias":None}
        try: self._load()
        except Exception as e: self.last["status"]="error"; self.last["error"]=str(e)[:240]

    def _load(self):
        # The research repository is private, so the trained artifact is bundled
        # with this deployment in deterministic chunks. This removes all runtime
        # GitHub authentication/download dependencies.
        parts_dir=Path(__file__).resolve().parent/"data"/"v3_model_parts"
        parts=sorted(parts_dir.glob("part_*.txt"))
        if parts:
            encoded="".join(p.read_text() for p in parts).strip()
        elif MODEL_PATH.exists() and MODEL_PATH.stat().st_size>=1000:
            encoded=MODEL_PATH.read_text().strip()
        else:
            raise RuntimeError("bundled V3 model parts are missing")
        raw=base64.b64decode(encoded)
        self.bundle=pickle.loads(raw)
        if self.bundle.get("version")!=3: raise RuntimeError("wrong V3 artifact")
        if self.bundle.get("features")!=FEATURES: raise RuntimeError(f"V3 feature mismatch model={self.bundle.get('features')} live={FEATURES}")

    @staticmethod
    def _features(rows):
        cols=["ts","open","high","low","close","volume","close_ts","quote_vol","trades","taker_base","taker_quote","ignore"]
        d=pd.DataFrame(rows,columns=cols)
        for c in ["ts","open","high","low","close","volume","trades","taker_base"]: d[c]=pd.to_numeric(d[c],errors="coerce")
        c=d.close; v=d.volume; tb=d.taker_base; tr=d.trades
        rs=lambda x,n:x.rolling(n,min_periods=n).sum()
        x=pd.DataFrame(index=d.index)
        for n in (2,3,5,10,15,30,60,120): x[f"move{n}"]=c-c.shift(n)
        x["accel5_15"]=(c-c.shift(5))-(c.shift(5)-c.shift(10))
        x["accel15_30"]=(c-c.shift(15))-(c.shift(15)-c.shift(30))
        for n in (5,15,30,60):
            vn=rs(v,n).replace(0,np.nan); x[f"imb{n}"]=2*rs(tb,n)/vn-1
        x["imb_accel"]=x["imb5"]-x["imb30"]
        v60=rs(v,60).replace(0,np.nan); t60=rs(tr,60).replace(0,np.nan)
        x["vol5ratio"]=rs(v,5)/v60*12; x["vol15ratio"]=rs(v,15)/v60*4
        x["trade5ratio"]=rs(tr,5)/t60*12; x["trade15ratio"]=rs(tr,15)/t60*4
        for n in (15,60,120):
            hi=d.high.rolling(n,min_periods=n).max(); lo=d.low.rolling(n,min_periods=n).min()
            x[f"range{n}"]=hi-lo; x[f"pos{n}"]=(c-lo)/(hi-lo).replace(0,np.nan)-.5
        ret=c.pct_change()
        x["rv15"]=ret.rolling(15,min_periods=15).std()*1e4; x["rv60"]=ret.rolling(60,min_periods=60).std()*1e4
        x["rv_ratio"]=x.rv15/x.rv60.replace(0,np.nan)
        body=d.close-d.open; x["body5"]=body.rolling(5,min_periods=5).sum()
        x["upperwick5"]=(d.high-np.maximum(d.open,d.close)).rolling(5,min_periods=5).sum()
        x["lowerwick5"]=(np.minimum(d.open,d.close)-d.low).rolling(5,min_periods=5).sum()
        sec=(d.ts//1000)%900
        x["round_sin"]=np.sin(2*np.pi*sec/900); x["round_cos"]=np.cos(2*np.pi*sec/900)
        row=x.iloc[-1][FEATURES].replace([np.inf,-np.inf],np.nan)
        if row.isna().any(): raise RuntimeError("warming 120s feature history")
        return row.to_numpy(float).reshape(1,-1), float(c.iloc[-1]), int(d.ts.iloc[-1])

    def _prob(self,target,X):
        z=self.bundle["models"][target]; raw=z["model"].predict_proba(X)[:,1]
        return float(z["calibrator"].predict(raw)[0])

    @staticmethod
    def _read(p,threshold):
        conf=max(p,1-p); direction="UP" if p>=.5 else "DOWN"
        return {"direction":direction,"probability_up":round(p,4),"confidence":round(conf,4),"actionable":bool(conf>=threshold)}

    @staticmethod
    def _bars_from_trades(trades, now_ms):
        by={}
        for t in trades:
            sec=int(t["T"])//1000; p=float(t["p"]); q=float(t["q"])
            b=by.setdefault(sec,{"open":p,"high":p,"low":p,"close":p,"volume":0.0,"trades":0,"taker_base":0.0})
            b["high"]=max(b["high"],p); b["low"]=min(b["low"],p); b["close"]=p; b["volume"]+=q; b["trades"]+=1
            if not bool(t.get("m",False)): b["taker_base"]+=q
        end=now_ms//1000; start=end-139; rows=[]; prev=None
        for sec in range(start,end+1):
            b=by.get(sec)
            if b is None:
                if prev is None: continue
                b={"open":prev,"high":prev,"low":prev,"close":prev,"volume":0.0,"trades":0,"taker_base":0.0}
            prev=b["close"]
            rows.append([sec*1000,b["open"],b["high"],b["low"],b["close"],b["volume"],sec*1000+999,0,b["trades"],b["taker_base"],0,0])
        return rows

    def _fetch_trades(self):
        # Primary: exact Binance.com BTCUSDT 1-second klines, matching the V3
        # historical feature source. Render may geo-block this endpoint, so a
        # clearly labelled Binance.US 1-minute proxy remains as paper fallback.
        now_ms=int(time.time()*1000)
        try:
            r=self.session.get(KLINES_URL,params={"symbol":"BTCUSDT","interval":"1s","limit":140},timeout=8)
            r.raise_for_status(); ks=r.json()
            if isinstance(ks,list) and len(ks)>=125:
                self._active_source="Binance.com BTCUSDT native 1s bars"
                self._source_exact=True
                return ks[-140:]
        except Exception as e:
            self._primary_error=str(e)[:160]

        r=self.session.get(US_KLINES_URL,params={"symbol":"BTCUSDT","interval":"1m","limit":4},timeout=8)
        r.raise_for_status(); ks=r.json()
        if not isinstance(ks,list) or len(ks)<3:
            raise RuntimeError(f"Binance.US fallback returned {len(ks) if isinstance(ks,list) else 'non-list'} klines; primary={getattr(self,'_primary_error','unavailable')}")
        rows=[]
        for k in ks:
            ot=int(k[0]); o=float(k[1]); h=float(k[2]); lo=float(k[3]); c=float(k[4]); vol=float(k[5]); trades=float(k[8]); tb=float(k[9])
            for i in range(60):
                frac=(i+1)/60.0; px=o+(c-o)*frac
                hi=max(px, h if i==30 else px); low=min(px, lo if i==30 else px)
                rows.append([ot+i*1000,px,hi,low,px,vol/60.0,ot+i*1000+999,0,trades/60.0,tb/60.0,0,0])
        self._active_source="Binance.US BTCUSDT 1m → 1s PAPER PROXY"
        self._source_exact=False
        return [x for x in rows if x[0] <= now_ms][-140:]

    def snapshot(self):
        now=time.time()
        if now-self.last_fetch<5: return self.last
        self.last_fetch=now
        try:
            if self.bundle is None: self._load()
            rows=self._fetch_trades()
            if len(rows)<125: raise RuntimeError(f"only {len(rows)} reconstructed one-second bars")
            X,price,ts=self._features(rows)
            p15=self._prob("dir15",X); p30=self._prob("dir30",X); p60=self._prob("dir60",X)
            up30=self._prob("up30_before_dn10",X); dn30=self._prob("dn30_before_up10",X)
            n15=self._read(p15,.80); n30=self._read(p30,.85); n60=self._read(p60,.85)
            scalp={"up30_before_down10":round(up30,4),"down30_before_up10":round(dn30,4),"research_only":True}
            state=n15["direction"] if n15["actionable"] else "WAIT"
            self.last={"version":"v3","mode":"paper","status":"live","source":getattr(self,"_active_source","unknown"),"source_exact_training_parity":bool(getattr(self,"_source_exact",False)),"source_price":round(price,2),"source_ts":ts,"age_seconds":max(0,round(now-ts/1000,1)),"state":state,"next_15":n15,"next_30":n30,"next_60":n60,"scalp_bias":scalp,"validation":{"dir15_threshold":.80,"dir15_holdout_accuracy":.850138,"dir15_holdout_coverage":.084822},"note":("Native Binance.com 1s input matches the historical bar source; still paper validation, not a Kalshi win-rate." if getattr(self,"_source_exact",False) else "LIVE INPUT IS A 1-MINUTE-TO-1-SECOND PROXY; historical accuracy must not be applied to these proxy predictions. Paper research only.")}
            if not self._reported_live:
                print(f"[V3] LIVE source={self.last.get('source')} exact={self.last.get('source_exact_training_parity')} price={self.last.get('source_price')} p15={self.last.get('next_15')}", flush=True)
                self._reported_live=True
        except Exception as e:
            self.last={**self.last,"status":"error","error":str(e)[:240],"age_seconds":None}
        return self.last

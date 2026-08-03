"""Expert-review video for spell-card detection: activation, cost, hit vs whiff.

    python -m scripts.spell_video --out /root/spell.mp4

Each clip is one detected spell-card window, captioned with the frame range, the
inferred cost in card slots, damage dealt during the window, and the resulting
HIT/WHIFF verdict. A black separator with the replay name precedes every clip --
the previous review video stitched clips from different matches with no break,
and two reviewers correctly read the character change as a health misread.
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sokubot.data.hud import (read_trace, damage_events, spellcard_events,
                              P1_CARD_X, P2_CARD_X, CARD_ROWS,
                              P1_SPIRIT_X, P2_SPIRIT_X, SPIRIT_N,
                              SPIRIT_CY_EVEN, SPIRIT_CY_ODD)
W = H = 480; PANEL = 118; FPS = 60

def decode(path):
    p = subprocess.run(["ffmpeg","-v","error","-i",path,"-f","rawvideo","-pix_fmt","rgb24","-"],
                       capture_output=True)
    n = len(p.stdout)//(W*H*3)
    return np.frombuffer(p.stdout[:n*W*H*3],dtype=np.uint8).reshape(n,H,W,3)

def draw(frame, t, d_foe, i, who, a, b, cost, dmg, replay, sep=False):
    img = Image.new("RGB",(W,H+PANEL),(10,9,14))
    if not sep:
        img.paste(Image.fromarray(frame),(0,0))
        dr = ImageDraw.Draw(img)
        cx = P1_CARD_X if who==1 else P2_CARD_X
        dr.rectangle([cx[0],CARD_ROWS[0],cx[1],CARD_ROWS[1]],outline=(255,200,60))
        sx = P1_SPIRIT_X if who==1 else P2_SPIRIT_X
        pitch=(sx[1]-sx[0])/SPIRIT_N
        for k in range(SPIRIT_N):
            ccx=int(sx[0]+pitch*(k+0.5)); ccy=SPIRIT_CY_EVEN if k%2==0 else SPIRIT_CY_ODD
            dr.rectangle([ccx-6,ccy-5,ccx+6,ccy+5],outline=(90,170,255))
    dr = ImageDraw.Draw(img)
    y0=H+6
    verdict = "HIT" if dmg>0.02 else "WHIFF"
    col = (110,220,150) if verdict=="HIT" else (235,110,110)
    dr.text((8,y0), f"P{who} SPELL CARD   {replay}   frames {a}-{b}  ({(b-a)/60:.1f}s)", fill=(217,164,65))
    dr.text((8,y0+16), f"inferred cost: {cost} card slots     damage dealt: {dmg:.3f}", fill=(230,225,240))
    dr.text((8,y0+32), f"verdict: {verdict}", fill=col)
    dr.text((8,y0+52), "CHECK: was a card actually cast here? is the cost right?", fill=(150,145,165))
    dr.text((8,y0+66), "       did it hit or whiff? did the window start/end correctly?", fill=(150,145,165))
    if not sep:
        prog=min(1.0,max(0.0,(i-a)/max(1,(b-a))))  # clip starts before the window
        dr.rectangle([8,y0+88,W-8,y0+98],outline=(70,62,88))
        dr.rectangle([8,y0+88,8+int((W-16)*prog),y0+98],fill=(140,110,200))
        cards = (t.cards1 if who==1 else t.cards2)[i]
        sp = (t.spirit1 if who==1 else t.spirit2)[i]
        dr.text((300,y0+32), f"cards lit {cards:.2f}  spirit {sp:.2f}", fill=(180,175,195))
    return np.asarray(img)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",type=Path,default=Path("/root/corpus"))
    ap.add_argument("--out",type=Path,default=Path("/root/spell.mp4"))
    ap.add_argument("--replays",type=int,default=10)
    ap.add_argument("--seconds",type=float,default=175.0)
    ap.add_argument("--target-mb",type=float,default=23.0)
    a=ap.parse_args()
    rows=[json.loads(l) for l in (a.corpus/"manifest.jsonl").read_text().splitlines()][7:7+a.replays]
    out=[]
    for r in rows:
        fr=decode(r["video"]); t=read_trace(fr,smooth=3)
        for who in (1,2):
            df=damage_events(t, 2 if who==1 else 1)
            for (s,e,cost,dmg) in spellcard_events(t,who):
                if e-s < 30: continue
                sepf=draw(None,t,df,s,who,s,e,cost,dmg,r["replay_id"][:12],sep=True)
                out.extend([sepf]*30)                      # 0.5 s black title card
                lo,hi=max(0,s-30),min(len(fr),e+30)
                for k in range(lo,hi):
                    out.append(draw(fr[k][::-1],t,df,k,who,s,e,cost,dmg,r["replay_id"][:12]))
                if len(out)/FPS>=a.seconds: break
            if len(out)/FPS>=a.seconds: break
        if len(out)/FPS>=a.seconds: break
    secs=len(out)/FPS; kbps=int((a.target_mb*8192)/max(secs,1)*0.92)
    print(f"{len(out)} frames = {secs:.0f}s -> {kbps} kbit/s",flush=True)
    p=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
        "-s",f"{W}x{H+PANEL}","-r",str(FPS),"-i","-","-c:v","libx264","-preset","slow",
        "-b:v",f"{kbps}k","-maxrate",f"{int(kbps*1.3)}k","-bufsize",f"{kbps*2}k",
        "-pix_fmt","yuv420p",str(a.out)],stdin=subprocess.PIPE)
    for f in out: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    print(f"wrote {a.out} - {secs:.0f}s, {a.out.stat().st_size/1e6:.1f} MB")

if __name__=="__main__": main()

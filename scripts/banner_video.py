"""Verification video for spell-card detection via the name banner.

Shows only detected windows, each preceded by a title card, with the search band
outlined and the frame-by-frame banner state marked. The question for a viewer is
narrow: was a spell card actually cast here, and does the window match its
duration?
"""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from sokubot.data.hud import (read_trace, damage_events, spellcard_events,
                              name_banner, NAME_BAND, P1_NAME_X, P2_NAME_X)
W=H=480; PANEL=104; FPS=60

def decode(p):
    r=subprocess.run(["ffmpeg","-v","error","-i",p,"-f","rawvideo","-pix_fmt","rgb24","-"],
                     capture_output=True)
    n=len(r.stdout)//(W*H*3)
    return np.frombuffer(r.stdout[:n*W*H*3],dtype=np.uint8).reshape(n,H,W,3)

def draw(fr, on, i, who, a, b, dmg, rid, sep=False):
    img=Image.new("RGB",(W,H+PANEL),(10,9,14)); dr=ImageDraw.Draw(img)
    if not sep:
        img.paste(Image.fromarray(fr),(0,0)); dr=ImageDraw.Draw(img)
        x0,x1=P1_NAME_X if who==1 else P2_NAME_X
        col=(255,70,70) if on[i] else (90,80,100)
        dr.rectangle([x0,NAME_BAND[0],x1,NAME_BAND[1]],outline=col,width=2)
    y0=H+6
    dr.text((8,y0), f"P{who} SPELL CARD?   {rid}   frames {a}-{b}  ({(b-a)/60:.1f}s)",fill=(217,164,65))
    dr.text((8,y0+16), f"damage during window: {dmg:.3f}   ->  {'HIT' if dmg>0.02 else 'WHIFF'}",
            fill=(110,220,150) if dmg>0.02 else (235,110,110))
    dr.text((8,y0+36), "RED BOX = name banner detected on this frame",fill=(200,120,120))
    dr.text((8,y0+52), "CHECK: was a card actually cast? does the window match its length?",
            fill=(150,145,165))
    if not sep:
        prog=min(1.,max(0.,(i-a)/max(1,b-a)))
        dr.rectangle([8,y0+76,W-8,y0+86],outline=(70,62,88))
        dr.rectangle([8,y0+76,8+int((W-16)*prog),y0+86],fill=(140,110,200))
    return np.asarray(img)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus",type=Path,default=Path("/root/corpus"))
    ap.add_argument("--out",type=Path,default=Path("/root/banner.mp4"))
    ap.add_argument("--replays",type=int,default=8)
    ap.add_argument("--seconds",type=float,default=170.)
    ap.add_argument("--target-mb",type=float,default=23.)
    a=ap.parse_args()
    rows=[json.loads(l) for l in (a.corpus/"manifest.jsonl").read_text().splitlines()][7:7+a.replays]
    out=[]
    for r in rows:
        fr=decode(r["video"]); t=read_trace(fr,smooth=3)
        for who in (1,2):
            on=name_banner(fr,who)
            for (s,e,dmg) in spellcard_events(fr,t,who):
                out.extend([draw(None,on,s,who,s,e,dmg,r["replay_id"][:12],sep=True)]*24)
                lo,hi=max(0,s-24),min(len(fr),e+24)
                step=max(1,(hi-lo)//360)          # cap very long cards
                for k in range(lo,hi,step):
                    out.append(draw(fr[k][::-1],on,k,who,s,e,dmg,r["replay_id"][:12]))
                if len(out)/FPS>=a.seconds: break
            if len(out)/FPS>=a.seconds: break
        if len(out)/FPS>=a.seconds: break
    secs=len(out)/FPS; kbps=int((a.target_mb*8192)/max(secs,1)*.92)
    print(f"{len(out)} frames = {secs:.0f}s -> {kbps} kbit/s",flush=True)
    p=subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
      "-s",f"{W}x{H+PANEL}","-r",str(FPS),"-i","-","-c:v","libx264","-preset","slow",
      "-b:v",f"{kbps}k","-maxrate",f"{int(kbps*1.3)}k","-bufsize",f"{kbps*2}k",
      "-pix_fmt","yuv420p",str(a.out)],stdin=subprocess.PIPE)
    for f in out: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    print(f"wrote {a.out} - {secs:.0f}s, {a.out.stat().st_size/1e6:.1f} MB")

if __name__=="__main__": main()

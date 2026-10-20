"""生成界面设计稿；图中所有音频数据均为合成示意，不执行音频处理。"""
from pathlib import Path
import base64
import io
import math
from html import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H, SCALE = 1600, 1040, 2
BLUE, ORANGE = '#1685f8', '#f47729'
TEXT, MUTED, BORDER = '#263549', '#728094', '#d8e1eb'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


class Canvas:
    def __init__(self):
        self.image = Image.new('RGB', (W*SCALE, H*SCALE), '#f2f5f9')
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                    '<rect width="1600" height="1040" fill="#f2f5f9"/>']

    def rect(self, x, y, w, h, fill='white', stroke=None, radius=6):
        box = tuple(round(v*SCALE) for v in (x, y, x+w, y+h))
        def rounded(bounds, r, color):
            a,b,d,e=bounds
            if r<=0:
                self.draw.rectangle(bounds,fill=color)
                return
            self.draw.rectangle((a+r,b,d-r,e),fill=color)
            self.draw.rectangle((a,b+r,d,e-r),fill=color)
            for cx,cy in [(a+r,b+r),(d-r,b+r),(a+r,e-r),(d-r,e-r)]:
                self.draw.ellipse((cx-r,cy-r,cx+r,cy+r),fill=color)
        rounded(box,radius*SCALE,stroke or fill)
        if stroke:
            rounded((box[0]+SCALE,box[1]+SCALE,box[2]-SCALE,box[3]-SCALE),max(0,radius*SCALE-SCALE),fill)
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke or "none"}"/>')

    def line(self, pts, color, width=1):
        self.draw.line([(round(x*SCALE), round(y*SCALE)) for x, y in pts], fill=color, width=max(1, round(width*SCALE)))
        p = ' '.join(f'{x:.2f},{y:.2f}' for x,y in pts)
        self.svg.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>')

    def text(self, x, y, value, size=18, color=TEXT, bold=False, anchor='left'):
        font = ImageFont.truetype(BOLD if bold else FONT, size*SCALE, index=2)
        tw = (self.draw.textlength(value,font=font) if hasattr(self.draw,'textlength') else self.draw.textsize(value,font=font)[0])/SCALE
        xx = x-tw/2 if anchor == 'center' else x-tw if anchor == 'right' else x
        if hasattr(self.draw,'textbbox'):
            self.draw.text((round(xx*SCALE), round(y*SCALE)), value, font=font, fill=color, anchor='lt')
        else:
            ox,oy=font.getoffset(value)
            self.draw.text((round(xx*SCALE)-ox,round(y*SCALE)-oy),value,font=font,fill=color)
        # Pillow's lt aligns to the glyph bounding box; SVG uses a matching top baseline.
        self.svg.append(f'<text x="{xx:.2f}" y="{y+size*.88:.2f}" font-family="Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{escape(value)}</text>')

    def dot(self, x, y, radius, color, stroke=None):
        self.draw.ellipse(tuple(round(v*SCALE) for v in (x-radius,y-radius,x+radius,y+radius)),fill=color,outline=stroke,width=SCALE)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="{stroke or "none"}"/>')

    def bitmap(self, im, x, y, w, h):
        resampling=getattr(Image,'Resampling',Image)
        self.image.paste(im.resize((round(w*SCALE), round(h*SCALE)), resampling.BILINEAR), (round(x*SCALE), round(y*SCALE)))
        b=io.BytesIO(); im.save(b, format='PNG')
        self.svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" preserveAspectRatio="none"/>')

    def save(self):
        self.image.save(OUT/'界面设计.png')
        (OUT/'界面设计.svg').write_text('\n'.join(self.svg+['</svg>']), encoding='utf-8')


c=Canvas()


def button(x,y,w,label,primary=False,selected=False,disabled=False):
    color = BLUE if primary or selected else '#f6f8fb'
    c.rect(x,y,w,38,color,BLUE if primary or selected else BORDER,5)
    c.text(x+w/2,y+10,label,17,'white' if primary or selected else '#a0a9b5' if disabled else TEXT,selected,'center')


def panel(y,h,title):
    c.rect(24,y,420,h,'#ffffff',BORDER,6)
    c.rect(25,y+1,418,32,'#f4f7fb',radius=5)
    c.text(38,y+9,title,18,'#285a9a',True)


def check(x,y,label):
    c.rect(x,y,17,17,BLUE,radius=3)
    c.line([(x+4,y+9),(x+7,y+12),(x+13,y+5)],'white',2)
    c.text(x+24,y,label,16)


def slider(y,label,value,pct,unit=''):
    c.text(39,y,label,17)
    x=158; w=175; yy=y+10
    c.line([(x,yy),(x+w,yy)],'#d4dce6',5)
    c.line([(x,yy),(x+w*pct,yy)],BLUE,5)
    c.dot(x+w*pct,yy,7,'white','#95a8bd')
    c.rect(348,y-3,80,27,'#f9fbfd',BORDER,3)
    c.text(388,y+3,f'{value}{unit}',15,TEXT,anchor='center')


# Window header.
c.rect(0,0,W,52,'#e5ebf3',radius=0)
c.line([(26,30),(31,18),(37,34),(43,14),(49,29)],BLUE,3)
c.text(62,15,'人声音效台',22,TEXT,True)
c.rect(220,13,196,27,'#f5f8fc',radius=4)
c.text(318,19,'Python 版 · 界面设计稿',14,'#5e6f84',anchor='center')
for x in [1480,1528,1575]:
    if x==1480: c.line([(x-5,26),(x+5,26)],MUTED,2)
    elif x==1528: c.rect(x-5,21,10,10,'#e5ebf3',MUTED,0)
    else:
        c.line([(x-5,21),(x+5,31)],MUTED,1.5)
        c.line([(x-5,31),(x+5,21)],MUTED,1.5)

# Left controls, following the supplied MATLAB example.
panel(72,154,'音频采集')
button(38,116,122,'● 开始录音')
button(169,116,122,'■ 结束录音',disabled=True)
button(300,116,129,'导入 WAV')
c.text(39,168,'采样率：44100 Hz    时长：6.20 s',17)
c.text(39,198,'录音提示：先采集底噪，再开始说话',15,MUTED)

panel(238,160,'去噪与声音增强')
button(38,281,122,'一键去噪')
button(169,281,122,'人声增强')
check(309,292,'使用去噪')
button(38,330,122,'采集底噪')
c.text(174,341,'底噪已采集 · 1.50 s',16,'#378467')
slider(371,'音量增益','1.00',.5,'×')

panel(410,150,'一键音效预设')
for i,label in enumerate(['原声','磁性','娃娃音','大叔音','电话音','山谷回声','机械电音','自定义']):
    button(38+(i%4)*99,454+(i//4)*47,94,label,selected=label=='电话音')

panel(572,220,'参数调节')
c.text(428,583,'松开滑块后点击处理',13,MUTED,anchor='right')
for y,label,value,pct,unit in [
    (618,'播放倍率','1.00',.333,'×'),
    (646,'低频增益','0.0',.5,' dB'),
    (674,'中频增益','0.0',.5,' dB'),
    (702,'高频增益','0.0',.5,' dB'),
    (730,'回声延时','0.30',.3,' s'),
    (758,'回声强度','0.00',0,'')]:
    slider(y,label,value,pct,unit)

panel(804,148,'对比试听')
for i,label in enumerate(['原声','去噪','增强','音效']):
    x=45+i*99
    c.dot(x,855,7,'white',BLUE if i==3 else '#a6b2c1')
    if i==3: c.dot(x,855,3.5,BLUE)
    c.text(x+15,847,label,17)
button(38,878,122,'▶ 播放',primary=True)
button(169,878,122,'Ⅱ 暂停')
button(300,878,129,'■ 停止')
c.rect(39,930,270,8,'#e3e8ef',radius=4)
c.rect(39,930,91,8,BLUE,radius=4)
c.text(428,926,'2.10 / 6.20 s',15,MUTED,anchor='right')
button(24,970,152,'开始处理',primary=True)
button(186,970,98,'重置')
button(294,970,150,'保存音频')

# Six analysis plots.
c.rect(460,72,1116,880,'white',BORDER,6)
c.rect(461,73,1114,32,'#f4f7fb',radius=5)
c.text(476,81,'处理前后可视化对比',18,'#285a9a',True)
c.text(1557,83,'示意数据 · 演示语音.wav',15,MUTED,anchor='right')

rng=np.random.default_rng(20261020)
centers=np.array([.35,.62,1.25,1.87,2.2,2.84,3.6,4.12,4.6,5.2,5.72])
amps=np.array([.6,.8,.6,.95,.65,.76,.8,.65,.9,.7,.55])
t=np.linspace(0,6.2,2200)
env=sum(a*np.exp(-((t-m)/.10)**2/2) for m,a in zip(centers,amps))
carrier=np.sin(2*np.pi*43*t)+.43*np.sin(2*np.pi*113*t)+.21*rng.normal(size=len(t))
wave=.65*env*carrier+.025*rng.normal(size=len(t))
wave=np.clip(wave,-1,1)
processed=wave*.84


def axes(x,y,title,kind):
    w,h=472,(153 if kind!='spec' else 165)
    c.text(x+w/2,y-35,title,19,TEXT,True,'center')
    c.rect(x,y,w,h,'white','#afbbc9',radius=0)
    xmax=8000 if kind=='freq' else 6.2
    xticks=[0,2000,4000,6000,8000] if kind=='freq' else [0,1,2,3,4,5,6.2]
    for v in xticks:
        xx=x+w*v/xmax
        c.line([(xx,y),(xx,y+h)],'#e4eaf2',1)
        c.text(xx,y+h+9,str(v),13,MUTED,anchor='center')
    labels=[('1',0),('0.5',.25),('0',.5),('-0.5',.75),('-1',1)] if kind=='wave' else [('0',0),('-20',.25),('-40',.5),('-60',.75),('-80',1)] if kind=='freq' else [('8000',0),('6000',.25),('4000',.5),('2000',.75),('0',1)]
    for label,pos in labels:
        yy=y+h*pos
        c.line([(x,yy),(x+w,yy)],'#e4eaf2',1)
        c.text(x-10,yy-7,label,13,MUTED,anchor='right')
    c.text(x+w/2,y+h+31,'频率 (Hz)' if kind=='freq' else '时间 (s)',14,MUTED,anchor='center')
    c.text(x-4,y-19,'幅度' if kind=='wave' else '幅度 (dB)' if kind=='freq' else '频率 (Hz)',13,MUTED)
    return w,h


for x,result in [(526,False),(1070,True)]:
    color=ORANGE if result else BLUE
    prefix='电话音' if result else '原始语音'
    w,h=axes(x,155,prefix+' · 波形','wave')
    values=processed if result else wave
    c.line([(x+i/(len(values)-1)*w,155+h/2-v*h*.44) for i,v in enumerate(values)],color,.85)

    w,h=axes(x,402,prefix+' · 幅度频谱','freq')
    f=np.linspace(0,8000,850)
    orig=-17-47*(f/8000)**.65+rng.normal(0,3,len(f))
    if result:
        inside=(f>=300)&(f<=3400)
        vals=np.where(inside,orig-3,-73+rng.normal(0,1.4,len(f)))
    else: vals=orig
    vals=np.clip(vals,-79,-2)
    c.line([(x+i/(len(vals)-1)*w,402-v*h/80) for i,v in enumerate(vals)],color,1.05)


def heatmap(result):
    nt,nf=480,165
    ts=np.linspace(0,6.2,nt)[None,:]
    fs=np.linspace(8000,0,nf)[:,None]
    e=sum(a*np.exp(-((ts-m)/.115)**2/2) for m,a in zip(centers,amps))
    f0=150+24*np.sin(ts*2.1)
    harmonic=np.exp(-(np.sin(np.pi*fs/f0)/.23)**2)
    formants=sum(a*np.exp(-((fs-(freq+35*np.sin(ts*3)))/spread)**2/2) for freq,spread,a in [(550,260,1),(1500,400,.7),(2700,500,.42),(4500,850,.18)])
    power=.04+rng.random((nf,nt))*.065+e*(.08+.4*harmonic+.72*formants)*np.exp(-fs/9500)
    if result:
        power=np.where((fs>=300)&(fs<=3400),power,.018+rng.random((nf,nt))*.025)
    z=np.clip(power,0,1)
    stops=np.array([[10,20,92],[20,45,173],[0,130,235],[0,208,198],[158,232,88],[255,224,44],[255,116,33]],dtype=float)
    pos=z*(len(stops)-1); idx=np.minimum(pos.astype(int),len(stops)-2); frac=(pos-idx)[...,None]
    rgb=(stops[idx]*(1-frac)+stops[idx+1]*frac).astype('uint8')
    return Image.fromarray(rgb)


for x,result in [(526,False),(1070,True)]:
    w,h=axes(x,649,('电话音' if result else '原始语音')+' · 语谱图','spec')
    c.bitmap(heatmap(result),x+1,650,w-2,h-2)

# Common color scale: same for the two illustrative spectrograms.
gradient=np.zeros((12,570,3),dtype=np.uint8)
stops=np.array([[10,20,92],[20,45,173],[0,130,235],[0,208,198],[158,232,88],[255,224,44],[255,116,33]],float)
for i in range(570):
    p=i/569*(len(stops)-1); j=min(int(p),len(stops)-2); gradient[:,i]=stops[j]*(1-(p-j))+stops[j+1]*(p-j)
c.bitmap(Image.fromarray(gradient),746,893,570,12)
c.text(728,891,'弱',14,MUTED,anchor='right')
c.text(1333,891,'强',14,MUTED)
c.text(1031,918,'能量 (dB) · 两图共用色标',14,MUTED,anchor='center')
c.dot(480,989,5,'#26a578')
c.text(495,979,'状态：处理完成',20,'#1771cb',True)
c.text(1558,981,'当前音效：电话音  |  去噪 → 增强 → 音效',16,MUTED,anchor='right')
c.text(24,1020,'界面原型 v1.0 · 图形为合成示意，仅用于布局与交互设计',13,MUTED)
c.text(1576,1020,'1600 × 1040  /  参考图同款布局',13,MUTED,anchor='right')
c.save()
print('Generated:',OUT/'界面设计.png',OUT/'界面设计.svg')

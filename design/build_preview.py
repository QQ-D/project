"""生成界面设计稿；图中所有音频数据均为合成示意，不执行音频处理。"""
from pathlib import Path
import base64
import io
import math
from html import escape

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent
W, H, SCALE = 1920, 1200, 2
BLUE, ORANGE = '#1685f8', '#f47729'
TEXT, MUTED, BORDER = '#263549', '#728094', '#d8e1eb'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'


class Canvas:
    def __init__(self):
        self.image = Image.new('RGB', (W*SCALE, H*SCALE), '#f2f5f9')
        self.draw = ImageDraw.Draw(self.image)
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
                    f'<rect width="{W}" height="{H}" fill="#f2f5f9"/>']

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

    def save(self, stem='界面设计'):
        self.image.save(OUT/f'{stem}.png')
        (OUT/f'{stem}.svg').write_text('\n'.join(self.svg+['</svg>']), encoding='utf-8')


c=Canvas()


def button(x,y,w,label,primary=False,selected=False,disabled=False,height=38):
    color = BLUE if primary or selected else '#f6f8fb'
    c.rect(x,y,w,height,color,BLUE if primary or selected else BORDER,5)
    c.text(x+w/2,y+(height-18)/2,label,17,'white' if primary or selected else '#a0a9b5' if disabled else TEXT,selected,'center')


def panel(y,h,title):
    c.rect(24,y,480,h,'#ffffff',BORDER,6)
    c.rect(25,y+1,478,32,'#f4f7fb',radius=5)
    c.text(38,y+9,title,18,'#285a9a',True)


def check(x,y,label):
    c.rect(x,y,17,17,BLUE,radius=3)
    c.line([(x+4,y+9),(x+7,y+12),(x+13,y+5)],'white',2)
    c.text(x+24,y,label,16)


def slider(y,label,value,pct,unit=''):
    c.text(39,y,label,17)
    x=174; w=210; yy=y+10
    c.line([(x,yy),(x+w,yy)],'#d4dce6',5)
    c.line([(x,yy),(x+w*pct,yy)],BLUE,5)
    c.dot(x+w*pct,yy,7,'white','#95a8bd')
    c.rect(402,y-3,86,27,'#f9fbfd',BORDER,3)
    c.text(445,y+3,f'{value}{unit}',15,TEXT,anchor='center')


# Window header.
c.rect(0,0,W,52,'#e5ebf3',radius=0)
c.line([(26,30),(31,18),(37,34),(43,14),(49,29)],BLUE,3)
c.text(62,15,'智能人声音效台',22,TEXT,True)
c.rect(250,13,196,27,'#f5f8fc',radius=4)
c.text(348,19,'Python 版 · 设计原型 v2',14,'#5e6f84',anchor='center')
for x in [1800,1848,1895]:
    if x==1800: c.line([(x-5,26),(x+5,26)],MUTED,2)
    elif x==1848: c.rect(x-5,21,10,10,'#e5ebf3',MUTED,0)
    else:
        c.line([(x-5,21),(x+5,31)],MUTED,1.5)
        c.line([(x-5,31),(x+5,21)],MUTED,1.5)

c.rect(24,64,480,32,'#e6edf7',radius=5)
button(24,64,210,'基础工作台 · 必做',selected=True,height=32)
button(244,64,260,'扩展功能 · 选做',height=32)
c.text(524,73,'录音或导入 → 底噪去噪 → 人声增强 → 音效制作 → 对比与保存',17,'#5c6e86')
c.text(1896,73,'离线处理 · 无需连续麦克风实时处理',16,MUTED,anchor='right')

# All mandatory controls are directly visible on the main workspace.
panel(108,128,'01  音频采集与管理')
button(38,151,140,'● 开始录音')
button(190,151,140,'■ 结束录音',disabled=True)
button(342,151,146,'导入 WAV')
c.text(39,200,'采样率：44100 Hz    时长：6.20 s    单声道',16)
c.text(39,222,'先采底噪，再录人声；也可导入已有录音',13,MUTED)

panel(248,156,'02  环境底噪与一键去噪')
button(38,290,140,'采集环境底噪')
button(190,290,140,'选取噪声片段')
button(342,290,146,'一键去噪')
check(39,341,'使用去噪结果')
c.text(488,342,'底噪已采集 · 1.50 s',15,'#378467',anchor='right')
slider(373,'去噪强度','60',.6,' %')

panel(416,180,'03  声音增强')
check(356,424,'启用增强')
check(39,461,'峰值归一化')
c.text(488,461,'目标峰值：0.95',16,MUTED,anchor='right')
slider(494,'音量增益','1.00',.5,'×')
slider(529,'人声清晰度','50',.5,' %')
c.text(39,571,'清晰度：基于均衡滤波调整人声频段',14,MUTED)
button(360,560,128,'一键增强',height=28)

panel(608,132,'04  一键音效预设')
for i,label in enumerate(['原声','磁性','娃娃音','大叔音','电话音','山谷回声','自定义']):
    button(38+(i%4)*114,650+(i//4)*43,108,label,selected=label=='电话音',height=34)
c.text(434,704,'可手动微调',14,MUTED,anchor='center')

panel(752,218,'05  参数调节')
c.text(488,763,'滑块与数值框同步',13,MUTED,anchor='right')
for y,label,value,pct,unit in [
    (796,'播放倍率','1.00',.333,'×'),
    (824,'低频增益','0.0',.5,' dB'),
    (852,'中频增益','0.0',.5,' dB'),
    (880,'高频增益','0.0',.5,' dB'),
    (908,'回声延时','0.30',.3,' s'),
    (936,'回声强度','0.00',0,'')]:
    slider(y,label,value,pct,unit)

panel(982,130,'06  即时切换对比试听')
for i,label in enumerate(['原声','去噪','增强','音效']):
    x=45+i*114
    c.dot(x,1026,7,'white',BLUE if i==3 else '#a6b2c1')
    if i==3: c.dot(x,1026,3.5,BLUE)
    c.text(x+15,1018,label,17)
button(38,1046,140,'▶ 播放',primary=True,height=32)
button(190,1046,140,'Ⅱ 暂停',height=32)
button(342,1046,146,'■ 停止',height=32)
c.rect(39,1095,315,7,'#e3e8ef',radius=3)
c.rect(39,1095,107,7,BLUE,radius=3)
c.text(488,1089,'2.10 / 6.20 s',15,MUTED,anchor='right')
button(24,1126,174,'开始处理',primary=True)
button(210,1126,100,'重置')
button(322,1126,182,'保存所选阶段 WAV')

# Six analysis plots.
c.rect(520,108,1376,1004,'white',BORDER,6)
c.rect(521,109,1374,32,'#f4f7fb',radius=5)
c.text(536,117,'处理前后可视化对比',18,'#285a9a',True)
c.text(1876,119,'示意数据 · 演示语音.wav',15,MUTED,anchor='right')
c.text(544,157,'原声始终保留；右列随试听阶段切换',16,MUTED)
c.text(1876,157,'当前：音效 / 电话音     显示频段：0–8000 Hz',16,MUTED,anchor='right')

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
    w,h=585,(170 if kind!='spec' else 200)
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


for x,result in [(590,False),(1260,True)]:
    color=ORANGE if result else BLUE
    prefix='电话音' if result else '原始语音'
    w,h=axes(x,224,prefix+' · 波形','wave')
    values=processed if result else wave
    c.line([(x+i/(len(values)-1)*w,224+h/2-v*h*.44) for i,v in enumerate(values)],color,.85)

    w,h=axes(x,490,prefix+' · 幅度频谱','freq')
    f=np.linspace(0,8000,850)
    orig=-17-47*(f/8000)**.65+rng.normal(0,3,len(f))
    if result:
        inside=(f>=300)&(f<=3400)
        vals=np.where(inside,orig-3,-73+rng.normal(0,1.4,len(f)))
    else: vals=orig
    vals=np.clip(vals,-79,-2)
    c.line([(x+i/(len(vals)-1)*w,490-v*h/80) for i,v in enumerate(vals)],color,1.05)


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


for x,result in [(590,False),(1260,True)]:
    w,h=axes(x,760,('电话音' if result else '原始语音')+' · 语谱图','spec')
    c.bitmap(heatmap(result),x+1,761,w-2,h-2)

# Common color scale: same for the two illustrative spectrograms.
gradient=np.zeros((12,570,3),dtype=np.uint8)
stops=np.array([[10,20,92],[20,45,173],[0,130,235],[0,208,198],[158,232,88],[255,224,44],[255,116,33]],float)
for i in range(570):
    p=i/569*(len(stops)-1); j=min(int(p),len(stops)-2); gradient[:,i]=stops[j]*(1-(p-j))+stops[j+1]*(p-j)
c.bitmap(Image.fromarray(gradient),920,1042,570,12)
c.text(900,1040,'弱',14,MUTED,anchor='right')
c.text(1510,1040,'强',14,MUTED)
c.text(1205,1074,'能量 (dB) · 同一参考值、同一颜色范围',14,MUTED,anchor='center')
c.dot(540,1145,5,'#26a578')
c.text(555,1135,'状态：处理完成',20,'#1771cb',True)
c.text(1876,1137,'处理链路：原声 → 去噪 → 归一化 / 均衡增强 → 电话音',16,MUTED,anchor='right')
c.text(24,1180,'静态设计稿 v2 · 所有图形与数值均为示意，不代表功能已实现',13,MUTED)
c.text(1896,1180,'主工作台：全部基础任务可见  /  扩展功能另页展示',13,MUTED,anchor='right')
c.save()

# Second screen: optional requirements have explicit, separately marked entries.
c=Canvas()
c.rect(0,0,W,52,'#e5ebf3',radius=0)
c.text(24,15,'智能人声音效台',22,TEXT,True)
c.text(260,20,'Python 版 · 扩展功能设计',15,MUTED)
button(24,64,210,'基础工作台 · 必做',height=32)
button(244,64,260,'扩展功能 · 选做',selected=True,height=32)
c.text(530,73,'选做功能集中展示；不影响主工作台基础任务验收',17,MUTED)
c.text(1896,73,'静态原型 · 扩展尚未实现',16,MUTED,anchor='right')


def extension_card(x,title,subtitle):
    c.rect(x,130,608,974,'white',BORDER,7)
    c.rect(x+1,131,606,46,'#edf3fb',radius=6)
    c.text(x+18,145,title,21,'#285a9a',True)
    c.text(x+18,194,subtitle,16,MUTED)


def field(x,y,w,label):
    c.rect(x,y,w,36,'#f9fbfd',BORDER,4)
    dropdown=label.endswith(' ▾')
    c.text(x+12,y+10,label[:-2] if dropdown else label,16)
    if dropdown:
        c.line([(x+w-25,y+15),(x+w-19,y+21),(x+w-13,y+15)],MUTED,1.5)


def ext_slider(x,y,label,value,pct):
    c.text(x+20,y,label,17)
    c.line([(x+190,y+10),(x+445,y+10)],'#d4dce6',5)
    c.line([(x+190,y+10),(x+190+255*pct,y+10)],BLUE,5)
    c.dot(x+190+255*pct,y+10,7,'white','#95a8bd')
    field(x+465,y-5,122,value)


extension_card(24,'A  高级音效与音效串联','对当前音频继续加工，可与基础去噪、增强串联。')
field(44,231,568,'输入阶段：增强结果  ▾')
check(44,289,'多重回声')
check(324,289,'房间混响')
ext_slider(24,335,'回声次数','3 次',.3)
ext_slider(24,385,'混响干湿比','30 %',.3)
check(44,447,'动态压缩')
ext_slider(24,493,'压缩阈值','−18 dB',.55)
ext_slider(24,543,'压缩比','3 : 1',.3)
check(44,605,'机械电音 / 调制')
ext_slider(24,651,'调制频率','80 Hz',.25)
c.text(44,717,'音高与语速独立调整',18,'#285a9a',True)
ext_slider(24,760,'音高','+0 半音',.5)
ext_slider(24,810,'语速','1.00×',.333)
c.text(44,871,'串联顺序',17,TEXT,True)
field(44,902,568,'变调 / 变速 → 压缩 → 调制 → 回声 / 混响')
button(44,954,170,'调整处理顺序')
button(44,1027,270,'应用高级音效',primary=True)
button(326,1027,286,'试听扩展结果',disabled=True)

extension_card(656,'B  中文文字转语音（TTS）','输入中文生成语音，再送入已有音效模块。')
field(676,231,568,'语音来源：可用 TTS 引擎  ▾')
c.text(676,289,'中文文本',18,TEXT,True)
c.rect(676,326,568,196,'#f9fbfd',BORDER,5)
c.text(694,346,'欢迎来到智能人声音效台。',20)
c.text(694,385,'这是一段用于综合实验演示的配音。',20)
c.text(1226,492,'示例文本 · 40 字以内',14,MUTED,anchor='right')
ext_slider(656,559,'合成语速','1.00×',.333)
button(676,618,270,'生成语音',primary=True)
button(958,618,286,'停止生成',disabled=True)
c.text(676,687,'生成状态：尚未生成',18,MUTED)
c.rect(676,730,568,8,'#e3e8ef',radius=4)
c.text(676,775,'采样率：—     时长：—     输出格式：WAV',17,MUTED)
button(676,833,270,'试听生成语音',disabled=True)
button(958,833,286,'保存生成 WAV',disabled=True)
c.rect(676,905,568,86,'#f1f6fe',radius=5)
c.text(696,924,'生成语音 → 主工作台原声 → 风格音效',18,'#285a9a',True)
c.text(696,956,'可制作电话音、磁性音、回声等风格配音',16,MUTED)
button(676,1027,568,'送入主工作台制作配音',disabled=True)

extension_card(1288,'C  智能语音增强与方法对比','训练小型网络或调用预训练模型，降噪或去混响。')
for i,label in enumerate(['预训练模型','小型 BP','小型 CNN']):
    button(1308+i*190,231,178,label,selected=i==0)
field(1308,289,386,'模型文件：尚未加载')
button(1706,289,170,'选择本地模型',height=36)
field(1308,345,386,'训练 / 验证数据：尚未选择')
button(1706,345,170,'选择数据集',height=36)
field(1308,401,270,'训练轮数：20')
field(1590,401,286,'批大小：16')
button(1308,457,270,'开始训练',disabled=True)
button(1590,457,286,'查看训练日志')
c.text(1308,509,'训练参数仅在 BP / CNN 模式启用',15,MUTED)
field(1308,550,568,'任务：智能降噪 / 去混响  ▾')
button(1308,606,568,'运行智能增强',disabled=True)
c.text(1308,667,'状态：模型尚未加载，未运行增强',17,MUTED)
c.text(1308,719,'传统方法与智能方法对比',19,'#285a9a',True)
for y,left,right in [(762,'方法','SNR 提升 / 处理耗时'),(804,'传统谱减','— / —'),(846,'智能增强','— / —')]:
    c.rect(1308,y,568,40,'#f3f6fb' if y==762 else '#fafbfd',BORDER,2)
    c.text(1320,y+12,left,16)
    c.text(1860,y+12,right,16,MUTED,anchor='right')
c.text(1308,912,'SNR 仅用于有干净参考的测试；实录无参考时显示 —',15,MUTED)
c.text(1308,948,'点击对比后，在主工作台查看图形并切换试听',16,MUTED)
button(1308,1027,270,'传统 / 智能对比',disabled=True)
button(1590,1027,286,'保存智能结果',disabled=True)

c.text(24,1138,'扩展设计覆盖：高级音效、TTS、智能语音增强；可按时间选择实现，不计入基础功能完成状态。',19,'#285a9a',True)
c.text(24,1180,'静态设计稿 v2 · 所有控件均为拟定入口；TTS 引擎、模型和数据集尚未配置',14,MUTED)
c.save('扩展功能设计')
print('Generated main and optional screens:',OUT)

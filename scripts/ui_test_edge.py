import sys
from playwright.sync_api import sync_playwright
S=sys.argv[1]
cases=[('flac giong that','/home/van/deepfake-video/data/asvspoof2019_LA/LA/ASVspoof2019_LA_eval/flac/LA_E_5450657.flac'),
       ('flac giong gia','/home/van/deepfake-video/data/asvspoof2019_LA/LA/ASVspoof2019_LA_eval/flac/LA_E_9497931.flac'),
       ('video khong tieng FF++','/home/van/deepfake-video/data/ff/manipulated_sequences/FaceSwap/c23/videos/280_249.mp4'),
       ('file hong',S+'/bad.mp4')]
with sync_playwright() as p:
    b=p.chromium.launch(); pg=b.new_page(viewport={'width':1000,'height':1400}); errs=[]
    pg.on('pageerror', lambda e: errs.append(str(e)))
    for k,(name,f) in enumerate(cases):
        pg.goto('http://localhost:8000/'); pg.wait_for_function("!document.getElementById('ver').textContent.includes('Đang')")
        pg.set_input_files('#file', f); vd=pg.is_disabled('#doVideo'); pg.click('#go')
        pg.wait_for_function("!document.getElementById('go').disabled && !document.querySelector('.spin')", timeout=120000)
        v=pg.inner_text('#vbody') if pg.is_visible('#vcard') else '(an)'; a=pg.inner_text('#abody') if pg.is_visible('#acard') else '(an)'
        pg.screenshot(path=f'{S}/ui_shots/edge_{k}.png', full_page=True)
        print(f'== {name} | o video bi tat: {vd}\n VIDEO: {v[:200]!r}\n AUDIO: {a[:200]!r}')
    print('page errors', errs); b.close()

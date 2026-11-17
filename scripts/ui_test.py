import csv, sys, time, json
import pandas as pd
from playwright.sync_api import sync_playwright
S = sys.argv[1]; URL = 'http://localhost:8000/'
pick = pd.read_csv(f'{S}/ui_pick.csv'); rows = []
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={'width': 1000, 'height': 1400})
    errs = []; pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None); pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(URL); pg.wait_for_function("!document.getElementById('ver').textContent.includes('Đang')", timeout=30000)
    ver = pg.inner_text('#ver'); print('Header:', ver, flush=True)
    for i, r in pick.iterrows():
        pg.goto(URL); pg.wait_for_function("!document.getElementById('ver').textContent.includes('Đang')")
        pg.set_input_files('#file', r.path)
        t0 = time.time(); pg.click('#go')
        pg.wait_for_function("!document.getElementById('go').disabled && !document.querySelector('.spin')", timeout=180000)
        wall = time.time() - t0
        def card(sel):
            el = pg.query_selector(sel); txt = el.inner_text() if el else ''
            badge = pg.query_selector(sel + ' .badge'); big = pg.query_selector(sel + ' .big')
            return (badge.inner_text() if badge else ''), (big.inner_text() if big else ''), txt
        vb, vs, vt = card('#vbody'); ab, as_, at = card('#abody')
        has_cam = pg.query_selector('#vbody img.cam') is not None; has_spark = pg.query_selector('#vbody svg.spark') is not None
        pg.screenshot(path=f'{S}/ui_shots/{i:02d}.png', full_page=True)
        rows.append(dict(i=i, y=r.y, video_verdict=vb, video_score=vs, audio_verdict=ab, audio_score=as_, heatmap=has_cam, spark=has_spark,
                         wall_s=round(wall, 2), video_err='Lỗi' in vt, audio_err='Lỗi' in at, video_text=vt.replace('\n', ' | ')[:300], audio_text=at.replace('\n', ' | ')[:300]))
        print(i, r.y, vb, vs, '|', ab, as_, f'{wall:.1f}s', flush=True)
    b.close()
pd.DataFrame(rows).to_csv(f'{S}/ui_results.csv', index=False)
print('console errors:', errs[:10])

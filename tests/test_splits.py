"""Kiem tra ro ri du lieu va tap test co dinh. Khong can du lieu hay mo hinh, chay duoc tren may sach."""
import csv, json, os
from collections import Counter
from conftest import ROOT

CFG = os.path.join(ROOT, 'configs')


def _split_ids():
    s = {k: json.load(open(os.path.join(CFG, 'splits', f'{k}.json'))) for k in ('train', 'val', 'test')}
    return {k: {i for pair in v for i in pair} for k, v in s.items()}


def test_ffpp_splits_disjoint():
    ids = _split_ids()
    assert not ids['train'] & ids['val']
    assert not ids['train'] & ids['test']
    assert not ids['val'] & ids['test']


def test_ffpp_c23_manifest_follows_splits():
    ids = _split_ids()
    rows = list(csv.DictReader(open(os.path.join(CFG, 'manifest_ffpp_c23.csv'))))
    assert len(rows) == 5000
    for r in rows:  # video gia "a_b" dung ca hai danh tinh a va b
        for part in r['video_id'].strip().split('_'):
            assert part in ids[r['split']], r


def test_ffpp_c40_is_test_only():
    ids = _split_ids()
    rows = list(csv.DictReader(open(os.path.join(CFG, 'manifest_ffpp_c40.csv'))))
    assert len(rows) == 700
    assert {r['split'] for r in rows} == {'test'}
    assert all(p in ids['test'] for r in rows for p in r['video_id'].strip().split('_'))


def test_celebdf_subset_fixed():
    rows = list(csv.DictReader(open(os.path.join(CFG, 'celebdf_test_subset.csv'))))
    assert len(rows) == 200
    assert len({r['path'] for r in rows}) == 200
    assert Counter(r['is_fake'] for r in rows) == {'0': 100, '1': 100}

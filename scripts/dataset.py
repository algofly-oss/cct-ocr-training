"""Synthetic examples and strict image/TXT to fast-plate-ocr CSV conversion."""
import argparse, csv, json, hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ALPHABET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
def convert(folder):
    folder=Path(folder); rows=[]
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {'.png','.jpg','.jpeg'}: continue
        text=path.with_suffix('.txt').read_text(encoding='utf8').rstrip('\r\n')
        if not text or len(text)>15 or any(c not in ALPHABET for c in text):
            raise ValueError(f'{path.name}: expected 1–15 uppercase letters/digits; no silent normalization')
        with Image.open(path) as im: im.verify()
        rows.append({'image_path':f'{folder.name}/{path.name}','plate_text':text})
    if not rows: raise ValueError('No image/TXT pairs')
    out=folder.parent/f'{folder.name}_annotations.csv'
    with out.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['image_path','plate_text']); w.writeheader();w.writerows(rows)
    return out,len(rows)
MARKER = 'cct-s-fictional-plates-v1'
def expected_files():
    return {'README.txt', 'train_annotations.csv', 'val_annotations.csv'} | {
        f'{split}/dummy_{i:02d}.{ext}' for split,n in [('train',16),('val',4)]
        for i in range(n) for ext in ['png','txt']}
def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_dummy(root):
    root=Path(root); manifest=root/'DUMMY_DATA.json'
    if root.is_symlink() or not manifest.is_file() or manifest.is_symlink():
        raise ValueError('Refusing existing folder without a regular DUMMY_DATA.json manifest')
    data=json.loads(manifest.read_text(encoding='utf8'))
    if data.get('marker')!=MARKER or set(data.get('sha256',{}))!=expected_files():
        raise ValueError('Invalid dummy-data manifest')
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual!=expected_files()|{'DUMMY_DATA.json'} or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Dummy folder contains unexpected, missing or linked files')
    for name,digest in data['sha256'].items():
        if file_hash(root/name)!=digest:
            raise ValueError(f'Modified dummy fixture preserved without overwrite: {name}')
    return data

def dummy(root):
    root=Path(root)
    if root.exists() or root.is_symlink():
        verify_dummy(root)
        print('Verified existing dummy data; no files changed')
        return
    root.mkdir(parents=True,exist_ok=False)
    for split,n,offset in [('train',16,0),('val',4,16)]:
        d=root/split;d.mkdir(parents=True,exist_ok=True)
        for i in range(n):
            text=f'AB{offset+i:03d}CD'; im=Image.new('RGB',(128,64),(242,242,235)); draw=ImageDraw.Draw(im)
            draw.rectangle((3,12,124,51),outline=(20,20,20),width=2)
            # Pillow's bundled font avoids external font downloads/licenses.
            font=ImageFont.load_default(size=19);box=draw.textbbox((0,0),text,font=font)
            draw.text(((128-box[2])/2,21),text,font=font,fill=(15,15,15))
            im.save(d/f'dummy_{i:02d}.png');(d/f'dummy_{i:02d}.txt').write_text(text+'\n')
        print(convert(d))
    (root/'README.txt').write_text('20 generated, fictional plates. Pipeline smoke data only; not a real accuracy benchmark. Image + text only.\n')
    data={'marker':MARKER,'sha256':{name:file_hash(root/name) for name in sorted(expected_files())}}
    with (root/'DUMMY_DATA.json').open('x',encoding='utf8') as f: json.dump(data,f,indent=2); f.write('\n')
    verify_dummy(root)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['dummy','convert']);p.add_argument('path');a=p.parse_args()
    if a.mode=='dummy':dummy(a.path)
    else:print(convert(a.path))

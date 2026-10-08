"""Build the two split-fix PowerPoint decks from the 0930 v-space template (stdlib only).

Deck A (42 slides): model5 with the original split (``sweep1008_control`` figures, left) vs the
fixed split (``sweep1008_splitfix`` figures, right); slides 1-21 v-space, 22-42 reconstruction.
Deck B (21 slides): native (left) vs model5 (right), both from the split-fix sweep, v-space.

The template is read, never written. Master, layouts, theme and the slide XML skeleton carry
over; only the 0918 sigmas are used (the template's 0925 sigmas are dropped). The output is
byte-for-byte reproducible from the figures. Run from anywhere; ``--check`` only re-verifies.
"""

import hashlib
import re
import shutil
import struct
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[2]
WRITEUPS = ROOT / "Claude Files" / "writeups"
TEMPLATE = WRITEUPS / "0930_vspace_native_vs_model5_all_sigma_bif_rgb.pptx"
FIGS = ROOT / "Simulated Data" / "Sweeps and Results" / "shift_sigma_sweep_bifurcation" / "1008"
CONTROL_FIGS = FIGS / "sweep1008_control_figs"
SPLITFIX_FIGS = FIGS / "figs"
OUT_A = WRITEUPS / "1008_vspace_recon_model5_original_vs_splitfix_bif.pptx"
OUT_B = WRITEUPS / "1008_vspace_native_vs_model5_splitfix_bif.pptx"

SIGMAS = [0.1, 0.5, 1.0, 2.0, 5.0, 7.5, 10.0]  # the 0918 sigmas (no 0925 sigmas)
SEEDS = [3, 4, 5]
DSEEDS = [1, 2, 3]

# Template geometry (EMU), measured from the template's slides.
PIC_W, PIC_H = 6095847, 2098743
COL_X = [0, 6095847]
ROW_Y = [561771, 2660514, 4759257]
COL_CENTRE = [COL_X[0] + PIC_W // 2, COL_X[1] + PIC_W // 2]
TITLE_CENTRE = 5099420 + 2137958 // 2
CHAR_W, PAD = 93000, 182880  # measured: 21 chars at 18 pt = 2137958 EMU incl. padding
FIXED_TS = (2026, 10, 8, 0, 0, 0)

SUB = "decipherseed = 1,2,3"
SUB_B = "decipherseed = 1,2,3 (train/val split fixed)"


def sigma_tag(sigma):
    return f"{sigma:g}"


def fig_path(root, arm_dir, kind, sigma, seed, dseed, model_name):
    name_sigma = f"{sigma}" if kind == "reconstruction" else sigma_tag(sigma)
    return (
        root / arm_dir / "bif" / f"sigma{sigma}" / kind
        / f"{kind}_sigma{name_sigma}_seed{seed}_{model_name}_bif_genes_decipherseed_{dseed}.png"
    )  # fmt: skip


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


def title_text(sigma, seed, suffix=""):
    return f"Sigma = {sigma_tag(sigma)}, Seed = {seed}{suffix}"


def read_template():
    zin = zipfile.ZipFile(TEMPLATE)
    parts = {n: zin.read(n) for n in zin.namelist()}
    order = zin.namelist()
    titles = {}
    for n in order:
        if re.fullmatch(r"ppt/slides/slide\d+\.xml", n):
            texts = re.findall(r"<a:t>(.*?)</a:t>", parts[n].decode())
            titles.setdefault(texts[0], n)
    return parts, order, titles


def pic_xml(idx, rid, x, y, cx, cy, descr):
    return (
        f'<p:pic><p:nvPicPr><p:cNvPr id="{idx + 1}" name="Picture {idx}" '
        f'descr="{escape(descr)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr>'
        f'<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/>'
        f'</a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{x}" y="{y}"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f"</p:spPr></p:pic>"
    )


def set_box(sp, centre, text):
    """Rewrite one template text box: new text, width from the text, centred on ``centre``."""
    w = len(text) * CHAR_W + PAD
    sp = re.sub(r'<a:off x="\d+"', f'<a:off x="{centre - w // 2}"', sp, count=1)
    sp = re.sub(r'<a:ext cx="\d+"', f'<a:ext cx="{w}"', sp, count=1)
    return sp


def slide_xml(skeleton, title, subtitle, left_hdr, right_hdr, pics, small_sub=False):
    """``skeleton`` = template slide XML; pics = [(png_path, x, y, cx, cy)] x6."""
    head, rest = skeleton.split("<p:pic>", 1)
    sps = re.findall(r"<p:sp>.*?</p:sp>", rest, flags=re.S)
    assert len(sps) == 3, len(sps)
    tail = rest[rest.rfind("</p:sp>") + len("</p:sp>") :]
    pic_blocks = "".join(
        pic_xml(i + 1, f"rId{i + 2}", x, y, cx, cy, p.name)
        for i, (p, x, y, cx, cy) in enumerate(pics)
    )
    s0 = set_box(sps[0], TITLE_CENTRE, max([title, subtitle], key=len))
    s0 = s0.replace("Sigma = 0.1, Seed = 3", escape(title)).replace(SUB, escape(subtitle))
    if small_sub:  # keep the longer subtitle clear of the column headers
        s0 = s0.replace(
            f'<a:rPr kumimoji="1" lang="en-US" altLang="ko-KR"/><a:t>{escape(subtitle)}',
            f'<a:rPr kumimoji="1" lang="en-US" altLang="ko-KR" sz="1400"/><a:t>{escape(subtitle)}',
        )
        s0 = re.sub(
            r'<a:ext cx="\d+"', f'<a:ext cx="{int(len(subtitle) * CHAR_W * 0.8) + PAD}"', s0, 1
        )
        w = int(len(subtitle) * CHAR_W * 0.8) + PAD
        s0 = re.sub(r'<a:off x="\d+"', f'<a:off x="{TITLE_CENTRE - w // 2}"', s0, count=1)
    s1 = set_box(sps[1], COL_CENTRE[0], left_hdr).replace("Base Decipher", escape(left_hdr))
    s2 = set_box(sps[2], COL_CENTRE[1], right_hdr).replace(
        "Decipher-BC (model5)", escape(right_hdr)
    )
    return head + pic_blocks + s0 + s1 + s2 + tail


def vspace_pics(left_paths, right_paths):
    pics = []
    for col, paths in ((0, left_paths), (1, right_paths)):
        for row, p in enumerate(paths):
            pics.append((p, COL_X[col], ROW_Y[row], PIC_W, PIC_H))
    return pics


def recon_pics(left_paths, right_paths):
    pics = []
    for col, paths in ((0, left_paths), (1, right_paths)):
        for row, p in enumerate(paths):
            w, h = png_size(p)
            cy = round(PIC_W * h / w)
            pics.append((p, COL_X[col], ROW_Y[row] + (PIC_H - cy) // 2, PIC_W, cy))
    return pics


def write_deck(out, slides, template_parts, template_order):
    """slides = [(xml, [png paths])]. Reuses every non-slide template part."""
    keep = [
        n for n in template_order
        if not re.fullmatch(r"ppt/slides/(_rels/)?slide\d+\.xml(\.rels)?", n)
        and not n.startswith("ppt/media/")
    ]  # fmt: skip
    pres = template_parts["ppt/presentation.xml"].decode()
    assert "sectionLst" not in pres
    ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{100 + i}"/>' for i in range(len(slides)))
    pres = re.sub(
        r"<p:sldIdLst>.*?</p:sldIdLst>", f"<p:sldIdLst>{ids}</p:sldIdLst>", pres, flags=re.S
    )
    prels = template_parts["ppt/_rels/presentation.xml.rels"].decode()
    prels = re.sub(r'<Relationship [^>]*relationships/slide"[^>]*/>', "", prels)
    new_rels = "".join(
        f'<Relationship Id="rId{100 + i}" Type="http://schemas.openxmlformats.org/officeDocument/'
        f'2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
        for i in range(len(slides))
    )
    prels = prels.replace("</Relationships>", new_rels + "</Relationships>")
    ctypes = template_parts["[Content_Types].xml"].decode()
    ctypes = re.sub(r'<Override PartName="/ppt/slides/slide\d+\.xml"[^>]*/>', "", ctypes)
    ctypes = ctypes.replace(
        "</Types>",
        "".join(
            f'<Override PartName="/ppt/slides/slide{i + 1}.xml" ContentType="application/'
            f'vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for i in range(len(slides))
        )
        + "</Types>",
    )
    app = template_parts["docProps/app.xml"].decode()
    app = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{len(slides)}</Slides>", app)
    app = re.sub(r"<Notes>\d+</Notes>", "<Notes>0</Notes>", app)
    app = re.sub(r"<HeadingPairs>.*?</TitlesOfParts>", "", app, flags=re.S)
    replaced = {
        "ppt/presentation.xml": pres.encode(),
        "ppt/_rels/presentation.xml.rels": prels.encode(),
        "[Content_Types].xml": ctypes.encode(),
        "docProps/app.xml": app.encode(),
    }
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:

        def put(name, data):
            zi = zipfile.ZipInfo(name, FIXED_TS)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zo.writestr(zi, data)

        for n in keep:
            put(n, replaced.get(n, template_parts[n]))
        img = 0
        for i, (xml, pngs) in enumerate(slides):
            rels = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="'
                'http://schemas.openxmlformats.org/package/2006/relationships"><Relationship '
                'Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                'relationships/slideLayout" Target="../slideLayouts/slideLayout7.xml"/>'
            )
            for j, p in enumerate(pngs):
                img += 1
                put(f"ppt/media/image{img}.png", Path(p).read_bytes())
                rels += (
                    f'<Relationship Id="rId{j + 2}" Type="http://schemas.openxmlformats.org/'
                    f'officeDocument/2006/relationships/image" Target="../media/image{img}.png"/>'
                )
            put(f"ppt/slides/_rels/slide{i + 1}.xml.rels", (rels + "</Relationships>").encode())
            put(f"ppt/slides/slide{i + 1}.xml", xml.encode())
    shutil.move(tmp, out)


def expected_decks():
    """Per deck: list of (title, subtitle, left_hdr, right_hdr, kind, left[3], right[3])."""
    a, b = [], []
    for kind in ("vspace", "reconstruction"):
        for sigma in SIGMAS:
            for seed in SEEDS:
                suffix = "" if kind == "vspace" else " — reconstruction"
                left = [
                    fig_path(CONTROL_FIGS, "model5", kind, sigma, seed, d, "model5") for d in DSEEDS
                ]
                right = [
                    fig_path(SPLITFIX_FIGS, "model5", kind, sigma, seed, d, "model5")
                    for d in DSEEDS
                ]
                a.append((title_text(sigma, seed, suffix), SUB, "model5, original split",
                          "model5, fixed split", kind, left, right))  # fmt: skip
    for sigma in SIGMAS:
        for seed in SEEDS:
            left = [
                fig_path(SPLITFIX_FIGS, "native_genes", "vspace", sigma, seed, d, "native")
                for d in DSEEDS
            ]
            right = [
                fig_path(SPLITFIX_FIGS, "model5", "vspace", sigma, seed, d, "model5")
                for d in DSEEDS
            ]
            b.append((title_text(sigma, seed), SUB_B, "Base Decipher", "Decipher-BC (model5)",
                      "vspace", left, right))  # fmt: skip
    return a, b


def build():
    parts, order, titles = read_template()
    for _, _, _, _, _, left, right in sum(expected_decks(), []):
        for p in left + right:
            assert p.exists(), p
    for sigma in SIGMAS:
        for seed in SEEDS:
            assert title_text(sigma, seed) in titles, title_text(sigma, seed)
    skeleton = parts[titles[title_text(SIGMAS[0], SEEDS[0])]].decode()
    for out, spec in zip((OUT_A, OUT_B), expected_decks()):
        slides = []
        for title, sub, lh, rh, kind, left, right in spec:
            pics = (vspace_pics if kind == "vspace" else recon_pics)(left, right)
            xml = slide_xml(skeleton, title, sub, lh, rh, pics, small_sub=(out == OUT_B))
            slides.append((xml, [p[0] for p in pics]))
        write_deck(out, slides, parts, order)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check(template_sha_before):
    """The six deck checks. Returns a list of (name, ok, detail)."""
    res = []
    spec_a, spec_b = expected_decks()
    decks = [(OUT_A, spec_a, 42), (OUT_B, spec_b, 21)]
    ok1 = ok2 = ok4 = True
    d1 = []
    n_pics = 0
    for out, spec, n in decks:
        z = zipfile.ZipFile(out)
        assert z.testzip() is None
        pres = z.read("ppt/presentation.xml").decode()
        prels = z.read("ppt/_rels/presentation.xml.rels").decode()
        rid_t = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="(slides/slide\d+\.xml)"', prels))
        rid_t.update(
            {
                a: b
                for b, a in re.findall(r'Target="(slides/slide\d+\.xml)"[^>]*?Id="(rId\d+)"', prels)
            }
        )
        order_rids = re.findall(r'<p:sldId id="\d+" r:id="(rId\d+)"/>', pres)
        ok1 &= len(order_rids) == n
        d1.append(f"{out.name}: {len(order_rids)} slides")
        for rid, (title, sub, lh, rh, kind, left, right) in zip(order_rids, spec):
            sname = rid_t[rid].split("/")[1]
            xml = z.read(f"ppt/slides/{sname}").decode()
            texts = [t.replace("&amp;", "&") for t in re.findall(r"<a:t>(.*?)</a:t>", xml)]
            ok1 &= texts == [title, sub, lh, rh]
            rels = z.read(f"ppt/slides/_rels/{sname}.rels").decode()
            rmap = dict(re.findall(r'Id="(rId\d+)"[^>]*?Target="\.\./media/(image\d+\.png)"', rels))
            pics = re.findall(
                r'r:embed="(rId\d+)"/>.*?<a:off x="(\d+)" y="(\d+)"/><a:ext cx="(\d+)" cy="(\d+)"/>',
                xml,
            )
            ok1 &= len(pics) == 6
            for (rid_i, x, y, cx, cy), src, col, row in zip(
                pics, left + right, [0, 0, 0, 1, 1, 1], [0, 1, 2] * 2
            ):
                n_pics += 1
                ok2 &= sha(z.read("ppt/media/" + rmap[rid_i])) == sha(src.read_bytes())
                x, y, cx, cy = map(int, (x, y, cx, cy))
                if kind == "vspace":
                    ok4 &= (x, y, cx, cy) == (COL_X[col], ROW_Y[row], PIC_W, PIC_H)
                else:
                    w, h = png_size(src)
                    ok4 &= x == COL_X[col] and cx == PIC_W
                    ok4 &= abs(cx / cy - w / h) / (w / h) < 0.01
                    ok4 &= y >= ROW_Y[row] and y + cy <= ROW_Y[row] + PIC_H
    res.append(("1 slide counts + title order", ok1, "; ".join(d1)))
    res.append(("2 embedded bytes == source PNG sha256", ok2, f"{n_pics} pictures"))
    res.append(("3 deck A left == template right (n/a: control is sweep1008_control)", None, "n/a"))
    res.append(("4 v-space 6095847x2098743 at template offsets; recon aspect <1%", ok4, ""))
    after = sha(TEMPLATE.read_bytes())
    res.append(("5 template sha256 unchanged", after == template_sha_before, after))
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    res.append(
        ("6 soffice render", None, soffice or "soffice not found: decks NOT visually rendered")
    )
    return res


if __name__ == "__main__":
    before = sha(TEMPLATE.read_bytes())
    print("template sha256 before:", before)
    if "--check" not in sys.argv:
        build()
    for name, ok, detail in check(before):
        print("PASS" if ok else ("N/A " if ok is None else "FAIL"), name, "|", detail)

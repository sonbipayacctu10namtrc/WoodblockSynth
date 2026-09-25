"""
Sinh notebook Kaggle tự chứa DUY NHẤT cho scan 3D mộc bản: notebooks/mocban_scan3d_kaggle.ipynb

  python notebooks/build_notebook.py

Một notebook chạy cả hai hướng trên mọi scan trong Kaggle Dataset:
  A. render ảnh giả ảnh chụp (pyrender, cần GPU/EGL)          <- src/core/mocban_render/
  B. ảnh tăng cường hình học, độc lập ánh sáng (CPU)          <- src/core/mocban_enhance/

Notebook KHÔNG ghi module ra file (%%writefile): mỗi hàm của hai module thành một cell code riêng, đứng ngay sau đoạn
giải thích chi tiết của nó (notebook_docs.py). Hai package .py vẫn là bản gốc cho các công cụ chạy local; builder tách
chúng tự động và kiểm tra:
  * ghép các cell hàm lại cho ra ĐÚNG cây cú pháp (AST) của các file gốc -> notebook không lệch với code local;
  * mọi hàm / lớp đều có giải thích -> thêm hàm mà quên giải thích thì build dừng;
  * biến của các cell chạy không trùng tên hàm / hằng số của thư viện -> không ghi đè nhầm.
"""
import ast
import json
import re
import sys
from pathlib import Path

import notebook_docs as DOCS

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "src" / "core"
_BANNER = re.compile(r"^# -{60,}\n# (\d+[a-z]?\. .+)\n# -{60,}\n", re.M)


def package_src(pkg):
    """Ghép các file của package (thứ tự theo FILES trong __init__.py) thành MỘT nguồn như module phẳng:
    import + shim đứng trước banner đầu của mỗi file gom lên đầu (bỏ trùng, bỏ `from .x import` nội bộ và
    `from __future__`), phần còn lại của từng file nối tiếp nhau. Trong notebook mọi hàm chung một namespace nên
    không cần import nội bộ."""
    d = CORE / pkg
    files = ast.literal_eval(re.search(r"^FILES = (\[.*\])$", (d / "__init__.py").read_text(encoding="utf-8"),
                                       re.M).group(1))
    plain, froms, other, rest = set(), {}, [], []    # import x | from x import a, b | lệnh khác (shim)
    for f in files:
        src = (d / f).read_text(encoding="utf-8")
        lines = src.splitlines(keepends=True)
        first = _BANNER.search(src)
        first_line = src[:first.start()].count("\n")
        for node in ast.parse(src).body:
            if node.lineno - 1 >= first_line:
                break
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue                                  # docstring của file
            if isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module != "__future__":
                    names = froms.setdefault(node.module, [])
                    names += [ast.unparse(a) for a in node.names if ast.unparse(a) not in names]
            elif isinstance(node, ast.Import):
                plain |= {ast.unparse(a) for a in node.names}
            else:
                k = node.lineno - 1                       # kèm comment sát phía trên (vd. giải thích shim)
                while k > 0 and lines[k - 1].lstrip().startswith("#"):
                    k -= 1
                text = "".join(lines[k:node.end_lineno])
                if text not in other:
                    other.append(text)
        rest.append("".join(lines[first_line:]))

    def group(std):
        mods = sorted({m.split()[0] for m in plain} | set(froms), key=lambda m: m.split(".")[0])
        out = []
        for m in mods:
            if (m.split(".")[0] in sys.stdlib_module_names) != std:
                continue
            out += [f"import {a}" for a in sorted(plain) if a.split()[0] == m]
            if m in froms:
                out.append(f"from {m} import {', '.join(froms[m])}")
        return "\n".join(out)

    header = "\n\n".join(x for x in (group(True), group(False), "".join(other).rstrip("\n")) if x)
    return header + "\n\n\n" + "\n\n".join(rest)


render_src = package_src("mocban_render")
enhance_src = package_src("mocban_enhance")

cells = []
LIB_CODE = []      # mã các cell thư viện (để kiểm AST)
EXEC_CODE = []     # mã các cell chạy (để kiểm trùng tên)


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s, lib=False):
    s = s.strip("\n")
    if not lib:
        s = re.sub(r"\b(?:mr|me)\.", "", s)          # cell chạy gọi thẳng tên hàm (không còn module mr / me)
        EXEC_CODE.append(s)
    else:
        LIB_CODE.append(s)
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})


# -----------------------------------------------------------------------------------------------------
# Tách module thành cell: mỗi hàm / lớp một cell, import + hằng số liền nhau gộp một cell
# -----------------------------------------------------------------------------------------------------


def _signature(node):
    if isinstance(node, ast.ClassDef):
        return f"Lớp `{node.name}`"
    a = node.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return f"`{node.name}({', '.join(names)})`"


def module_cells(src, docs, modname):
    """Sinh cell markdown + code cho một module. Trả danh sách tên hàm / lớp / biến cấp module."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    body = list(tree.body)
    prev_end = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        prev_end = body[0].end_lineno                 # docstring module -> đã có ở markdown, bỏ khỏi code
        body = body[1:]
    names, pending, pending_nodes = [], [], []

    def flush():
        if not pending:
            return
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in pending_nodes):
            key = f"import:{modname}"
        else:
            tgt = next(n for n in pending_nodes if isinstance(n, (ast.Assign, ast.AnnAssign)))
            t0 = tgt.targets[0] if isinstance(tgt, ast.Assign) else tgt.target
            key = f"const:{t0.id}"
        if key not in docs:
            raise SystemExit(f"thiếu giải thích cho {key!r} trong notebook_docs.py")
        md(("#### Import & shim" if key.startswith("import:") else "#### Hằng số") + "\n" + docs[key].strip("\n"))
        code("".join(pending), lib=True)
        pending.clear(); pending_nodes.clear()

    for node in body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            prev_end = node.end_lineno                # chỉ tác dụng trong 1 cell; Python 3.12 không cần
            continue
        start = (node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno) - 1
        gap = "".join(lines[prev_end:start])
        m = _BANNER.search(gap)
        if m:                                          # sang mục mới: đóng chunk hằng số, in tiêu đề mục
            flush()
            title = m.group(1)
            md(f"### {title}\n" + DOCS.SECTION.get(title, ""))
            gap = ""                                   # chú thích ngay sau banner đã nằm trong SECTION
        lead = "".join(l for l in gap.splitlines(keepends=True) if l.strip())   # chú thích sát trước node
        text = lead + "".join(lines[start:node.end_lineno])
        prev_end = node.end_lineno
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            flush()
            if node.name not in docs:
                raise SystemExit(f"thiếu giải thích cho hàm {node.name!r} ({modname}) trong notebook_docs.py")
            md(f"#### {_signature(node)}\n" + docs[node.name].strip("\n"))
            code(text, lib=True)
            names.append(node.name)
        else:
            pending.append(text)
            pending_nodes.append(node)
            for t in ast.walk(node):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    names.append(t.id)
                elif isinstance(t, ast.alias):
                    names.append((t.asname or t.name).split(".")[0])
    flush()
    return names


def _body_for_compare(src):
    """Các câu lệnh cấp module, bỏ docstring và `from __future__` (hai thứ notebook cố ý không đưa vào cell)."""
    body = [n for n in ast.parse(src).body if not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return body


# =====================================================================================================
# Giới thiệu, dữ liệu, cài đặt
# =====================================================================================================
md(r"""
# Scan 3D mộc bản → ảnh 2D (render giả ảnh chụp + tăng cường hình học)

Từ **model 3D scan thật** của khối mộc bản / khối in, sinh ảnh 2D để làm giàu dữ liệu huấn luyện OCR. Notebook chạy
cho **mọi scan** trong dataset đầu vào, theo hai hướng:

| Hướng | Làm gì | Cần |
|---|---|---|
| **A — render giả ảnh chụp** | rasterize bằng pyrender với texture, nhiều **góc chụp** + **điều kiện ánh sáng**, hậu kỳ giả camera (phơi sáng, WB, nhiễu, mờ, JPEG), tuỳ chọn vật che 2D | **GPU** (EGL) |
| **B — tăng cường hình học** | chiếu scan thành **ảnh độ sâu** ở góc nhìn bất kỳ rồi làm nổi nét khắc bằng 6 phương pháp (độ sâu cục bộ, pháp tuyến, độ cong, MSII, AO, exaggerated shading) — **không dùng đèn**, nên không phụ thuộc góc chiếu sáng | CPU |

```
scan ──> nạp 1 lần + soát mặt khắc (chung) ──┬─> A: sample_shot → pyrender → hậu kỳ camera → vật che → JPG + meta.json
                                             └─> B: project_depth → ảnh độ sâu → 6 phép tăng cường → PNG + batch_meta.json
```

**Mục lục**
1. Cài đặt
2. **Thư viện hàm** — mỗi hàm một cell, có giải thích chi tiết ngay trước: phần nạp scan & render, phần tăng cường
3. Cấu hình (mọi tham số ở một chỗ) + kiểm tra GPU
4. Dò scan → nạp + soát mặt khắc (mỗi scan nạp **một lần**, dùng lại cho A và B)
5. Phần A — render giả ảnh chụp
6. Phần B — tăng cường hình học
7. Đóng gói `outputs.zip`

Notebook không ghi file mã nguồn nào ra thư mục output: mọi hàm được định nghĩa thẳng trong notebook. Output chỉ gồm
ảnh, meta và `outputs.zip`.

**Settings:** Accelerator = **GPU** (T4/P100), Internet = **On** (để pip cài).
Không bật GPU thì phần A **tự bỏ qua**, phần B vẫn chạy đủ.

**Dữ liệu:** notebook chạy trên mọi scan có trong dataset đầu vào. Có scan mộc bản chữ Hán thì chỉ cần đổi dataset.
""")

md(r"""
## 0. Chuẩn bị dữ liệu: tải lên Kaggle Dataset
Kaggle **không** tự tải được scan từ trang nguồn (thường yêu cầu đăng nhập). Làm 1 lần:

1. Tải model về máy (định dạng gốc OBJ hoặc glTF/GLB) → file `.zip`.
2. Kaggle → **Create → New Dataset** → kéo các file `.zip` vào (Kaggle tự giải nén, kể cả zip lồng bên trong). Một
   dataset chứa được **nhiều scan**; **tên file mesh không được trùng nhau**.
3. Notebook này: panel phải **Add Input → Datasets → Your Datasets** → chọn dataset.
4. *(Tuỳ chọn)* thêm `manifest.csv` vào dataset để chỉ định tay mặt khắc (xem mục soát bên dưới).

Texture: GLB thường nhúng sẵn. OBJ scan thường có ảnh rời (vd. `textures/xxx_4K.jpg`) — notebook tự tìm ảnh có tên
trùng tiền tố với mesh ở thư mục mesh và các thư mục xung quanh (lên tới 3 cấp), bỏ qua normal map / roughness.
Texture chỉ dùng cho hướng A.
""")

md("## 1. Cài đặt\nChỉ cài gói **còn thiếu** (Kaggle có sẵn numpy / scipy / opencv / matplotlib), in tiến trình pip "
   "ngay khi chạy, và kiểm tra Internet trước để không treo im lặng.")

code(r"""
import os, sys, subprocess, platform, socket, importlib.util, importlib.metadata as imd
# EGL headless phải đặt TRƯỚC mọi lần import OpenGL/pyrender: PyOpenGL chọn nền tảng ngay lúc import lần đầu
os.environ["PYOPENGL_PLATFORM"] = "egl"
print(platform.platform(), sys.version)
!nvidia-smi -L || echo "Không có GPU -> phần A (pyrender/EGL) sẽ bị bỏ qua, phần B vẫn chạy"

def have(mod):
    return importlib.util.find_spec(mod) is not None

def dist_version(name):
    try:
        return imd.version(name)
    except imd.PackageNotFoundError:
        return None

def pip(*args):
    # KHÔNG capture: in từng dòng ngay khi pip chạy để thấy đang tải gì; timeout mạng ngắn để không treo im lặng
    cmd = [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "--timeout", "30", "--retries", "2", *args]
    print("$ pip install", " ".join(args), flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        if not line.startswith("Requirement already satisfied"):
            print("   ", line, end="", flush=True)
    if p.wait():
        raise SystemExit(f"pip lỗi: {args}")

# Kaggle có sẵn numpy / scipy / opencv / matplotlib / pillow: KHÔNG cài lại (cài opencv-python-headless mới nhất
# sẽ kéo nâng cấp numpy -> rất lâu và dễ vỡ môi trường). Chỉ cài phần còn thiếu.
need = [pkg for mod, pkg in [("trimesh", "trimesh"), ("pyglet", "pyglet"), ("freetype", "freetype-py"),
                             ("imageio", "imageio"), ("six", "six"), ("networkx", "networkx"),
                             ("cv2", "opencv-python-headless"), ("scipy", "scipy"), ("matplotlib", "matplotlib"),
                             ("PIL", "pillow")] if not have(mod)]
# pyrender 0.1.45 pin PyOpenGL==3.1.0 (lỗi glGenTextures trên Python mới) -> cài no-deps + PyOpenGL 3.1.7
if dist_version("PyOpenGL") != "3.1.7":
    need.append("PyOpenGL==3.1.7")
need_pyrender = not have("pyrender")
print("cần cài:", need + (["pyrender (--no-deps)"] if need_pyrender else []) or "không — đủ cả")

if need or need_pyrender:
    try:
        socket.create_connection(("pypi.org", 443), timeout=5).close()
    except OSError:
        raise SystemExit("Không kết nối được pypi.org: bật Settings → Internet = On rồi chạy lại cell này")
    if need:
        pip(*need)
    if need_pyrender:
        pip("--no-deps", "pyrender")

import trimesh, cv2, numpy, scipy
print("trimesh", trimesh.__version__, "| cv2", cv2.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| PyOpenGL", dist_version("PyOpenGL"), "| pyrender", dist_version("pyrender"))
""")

# =====================================================================================================
# Thư viện hàm
# =====================================================================================================
md(r"""
## 2. Thư viện hàm
Mỗi hàm là một cell, đứng ngay sau phần giải thích: **làm gì**, **đầu vào / đầu ra**, **cách làm** (công thức) và
**lưu ý** (những lỗi đã gặp khi chạy trên scan thật). Chạy tuần tự mọi cell trong mục này trước khi sang mục 3.
Mã nguồn giống hệt hai package `src/core/mocban_render/` và `src/core/mocban_enhance/` trong repo (mỗi mục = một file).
""")

md("## 2A. Nạp scan & render (`mocban_render/`)\nDùng chung cho soát mặt khắc, phần A và phần B.")
lib_names = module_cells(render_src, DOCS.RENDER, "mocban_render")

md("## 2B. Tăng cường hình học (`mocban_enhance/`)\nDùng cho phần B. Không cần GPU.")
lib_names += module_cells(enhance_src, DOCS.ENHANCE, "mocban_enhance")

# =====================================================================================================
# Cấu hình chung
# =====================================================================================================
md(r"""
## 3. Cấu hình
Mọi tham số ở một chỗ. Kết quả ghi vào `/kaggle/working/outputs/`: `soat/` (ảnh soát + `audit.csv`),
`A_render/`, `B_enhance/`, cuối cùng nén thành `outputs.zip`.

Cell này cũng thử khởi tạo pyrender và in **GPU mà OpenGL thật sự dùng**. Nếu in ra `llvmpipe` / `softpipe` thì EGL
đang render bằng CPU dù máy có GPU — phần A vẫn chạy nhưng chậm hơn nhiều.
""")

code(r"""
import time, json, math, shutil, csv
from pathlib import Path
import numpy as np, cv2, trimesh
import matplotlib.pyplot as plt

INPUT = Path("/kaggle/input")
OUT   = Path("/kaggle/working/outputs")
OUT_SOAT, OUT_A, OUT_B = OUT / "soat", OUT / "A_render", OUT / "B_enhance"
for d_ in (OUT_SOAT, OUT_A, OUT_B):
    d_.mkdir(parents=True, exist_ok=True)

# --- chung ---
TARGET_MM          = 200.0     # scale cạnh dài nhất của khối về N mm
RENDER_UNCONFIDENT = False     # False: bỏ qua scan chưa chắc mặt khắc (sửa manifest.csv rồi chạy lại)

# --- hướng A: render giả ảnh chụp (GPU) ---
RUN_A          = True          # False: bỏ hẳn phần A (hoặc tự bỏ khi không có GPU)
RENDER_W, RENDER_H = 1600, 1200  # độ phân giải ảnh render (giảm nếu chậm / hết RAM)
N_SHOTS        = 24            # số ảnh render MỖI scan
PRESET         = "mixed"       # mixed | topdown | handheld | raking | closeup
OCCLUDER_PROB  = 0.0           # >0 để thêm vật che 2D (tay/thước/lóa/bóng)

# --- hướng B: tăng cường hình học (CPU) ---
SCAN        = None             # scan để phân tích chi tiết (tên file mesh); None = scan đầu tiên
PX_MM       = None             # None = độ phân giải gốc của scan (đo khi nhìn thẳng mặt khắc)
FLIP_MIRROR = True             # mộc bản khắc NGƯỢC -> lật ngang cho chữ đọc xuôi; khối không có chữ: False
CROP_MM     = 50.0             # cạnh vùng zoom (mm), tự đặt vào chỗ nhiều chi tiết nhất
SIGMA0_PX   = 1.0              # chính quy hoá: xoá xung đạo hàm trên cạnh tam giác + nhiễu đo
CURV_SIGMA  = 1.5
MSII_RADII  = (3, 6, 12, 24, 48)
AO_N_AZ     = 16
AO_R_PX     = 24
LIGHT       = light_dir(35.0, 135.0)
ANGLES      = [(90, 0), (70, 25), (50, 25), (30, 25)]     # góc chiếu để xem
BATCH_ANGLES  = [(90, 0), (70, 25), (50, 25)]            # xuất hàng loạt cho mọi scan
BATCH_METHODS = ("depth", "ao", "exaggerated")
# tham số chuyển cho các hàm m_* (lưới so sánh + xuất hàng loạt) -> đổi cấu hình ở trên là mọi ảnh đổi theo
METHOD_KW = dict(ldir=LIGHT, radii_px=MSII_RADII, n_az=AO_N_AZ, radius_px=AO_R_PX, sigma0_px=SIGMA0_PX)

def show(img_or_path, title="", figsize=(15, 10), cmap=None):
    im = img_or_path if isinstance(img_or_path, np.ndarray) else \
        cv2.cvtColor(cv2.imread(str(img_or_path)), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(im, cmap=cmap, interpolation="nearest")   # nearest: mép nét chỉ vài px, nội suy sẽ moiré
    plt.title(title); plt.axis("off"); plt.show()

def flip(a):
    return a[:, ::-1] if FLIP_MIRROR else a

def save(img_rgb, fname):
    cv2.imwrite(str(OUT_B / fname), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    return OUT_B / fname

# pyrender/EGL có chạy được không (quyết định phần A và kiểu ảnh soát) + GPU nào đang render
GL = {}
try:
    _t = trimesh.creation.box(extents=[50, 50, 10]); _t.metadata["block_size_mm"] = [50, 50, 10]
    _b = PyrenderBackend(_t, 64, 64); GL = _b.gl_info(); _b.close()
    EGL_OK = True
except Exception as e:
    EGL_OK = False
    print("pyrender/EGL không chạy được (Accelerator = GPU?):", repr(e)[:200])
print(f"pyrender/EGL: {'OK' if EGL_OK else 'KHÔNG'} -> phần A {'chạy' if RUN_A and EGL_OK else 'BỎ QUA'}; phần B chạy")
if GL:
    print(f"OpenGL: {GL['renderer']} | {GL['vendor']} | {GL['version']}")
    if any(k in GL["renderer"].lower() for k in ("llvmpipe", "softpipe", "swiftshader", "software")):
        print("CẢNH BÁO: OpenGL đang render bằng CPU (không tới được driver GPU) -> phần A sẽ chậm hơn nhiều.")
""")

# =====================================================================================================
# Dò scan + nạp + soát mặt khắc (một lần)
# =====================================================================================================
md(r"""
## 4. Dò scan + manifest trong `/kaggle/input`
`manifest.csv` (tuỳ chọn, cột `mesh, texture, face, rot90`) ghi đè lựa chọn tự động cho từng scan — khoá là **tên file
mesh**. Có thể dùng thẳng `audit.csv` do notebook sinh ra (đã sửa) và đổi tên thành `manifest.csv`.
""")

code(r"""
meshes = sorted(p for p in INPUT.rglob("*") if p.suffix.lower() in MESH_EXT)
assert meshes, "Không thấy file mesh trong /kaggle/input — đã Add Input dataset chưa?"
mesh_names = [p.name for p in meshes]
assert len(set(mesh_names)) == len(mesh_names), \
    f"Trùng tên file mesh: {sorted({n for n in mesh_names if mesh_names.count(n) > 1})}"

man_files = sorted(INPUT.rglob("manifest.csv"))
manifest = read_manifest(man_files[0]) if man_files else {}
print("manifest:", man_files[0] if man_files else "không có (tự động toàn bộ)")
for p in meshes:
    print(f"  {p.relative_to(INPUT)}  ({p.stat().st_size // (1024 * 1024)} MB)  chỉ định: {manifest.get(p.name) or '-'}")
""")

md(r"""
## 4b. Nạp scan + soát mặt khắc (dùng chung cho A và B)
Mỗi scan được **nạp đúng một lần** (`prepare_scan`, tốn 10–20 s với scan 1,4 triệu tam giác), soát mặt khắc
(`audit_prepared`), xoay mặt khắc lên +Z (`orient_to_face`) rồi giữ trong bộ nhớ (`SCANS`) cho phần A và phần B dùng lại.

Ảnh soát: có GPU → render nhìn thẳng từng mặt ứng viên (khung **xanh** = tự chọn & tin cậy · **vàng** = cần xem ·
**tím** = chỉ định tay); không GPU → ảnh độ sâu cục bộ của mặt được chọn (trái) và mặt đối diện (phải).
Dòng in ra cũng cho biết **texture**: `rời <tên file>`, `nhúng sẵn`, hoặc `KHÔNG CÓ` (ảnh phần A sẽ ra màu xám).

Sửa khi sai: tải `soat/audit.csv`, đổi cột `face` (vd. `C+`) / `rot90` (0–3) / `texture` cho dòng sai, lưu thành
`manifest.csv` vào dataset, chạy lại.
""")

code(r"""
def depth_pair_image(mesh_o):
    # ảnh soát KHÔNG cần GPU: độ sâu cục bộ nhìn thẳng mặt được chọn (trái) và mặt đối diện (phải)
    L_ = max(mesh_o.bounding_box.extents[:2])
    tiles = []
    for elev in (90, -90):
        P = project_depth(mesh_o, elev, 0, px_mm=L_ / 400)
        loc_ = depth_map(P["depth"], P["px_mm"], hp_sigma_px=6.0)["local"]
        tile = colorize(norm01(loc_, sym=True), "coolwarm"); tile[~P["valid"]] = 40
        tiles.append(cv2.resize(tile, (480, int(480 * tile.shape[0] / tile.shape[1]))))
    hh_ = max(t.shape[0] for t in tiles)
    return np.hstack([np.pad(t, ((0, hh_ - t.shape[0]), (0, 8), (0, 0)), constant_values=255) for t in tiles])

SCANS = {}      # tên file mesh -> {"path", "mesh" (mặt khắc +Z, có texture), "info", "texture"}
for p in meshes:
    t0 = time.time()
    ov = manifest.get(p.name, {})
    m_obb = prepare_scan(p, TARGET_MM, ov.get("texture", "auto"))
    t_load = time.time() - t0
    info, img = audit_prepared(m_obb, ov, image=EGL_OK)
    mesh_o = orient_to_face(m_obb, info["face"], info["rot90"])
    mesh_o.metadata["face_info"] = info
    del m_obb
    if img is None:
        img = depth_pair_image(mesh_o)
    tex = mesh_o.metadata.get("source_texture")
    SCANS[p.name] = {"path": p, "mesh": mesh_o, "info": info, "texture": tex}
    tex_msg = (f"rời {Path(tex).name}" if tex else "nhúng sẵn") if info["has_texture"] else \
        "KHÔNG CÓ -> ảnh phần A sẽ ra màu xám (thêm ảnh texture vào dataset hoặc ghi cột texture trong manifest)"
    print(f"[{'OK ' if info['confident'] else 'XEM'}] {p.name}: mặt {info['face']} — {info['reason']} "
          f"| texture: {tex_msg} | nạp {t_load:.0f}s, tổng {time.time() - t0:.0f}s")
    cv2.imwrite(str(OUT_SOAT / f"{p.stem}_faces.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    show(img, p.name + ("" if EGL_OK else "   (trái: mặt được chọn — phải: mặt đối diện)"), figsize=(16, 6))

with open(OUT_SOAT / "audit.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["mesh", "texture", "face", "rot90", "confident", "reason", "ratio", "auto_face", "has_texture", "scores"])
    for name_, s_ in SCANS.items():
        i = s_["info"]
        w.writerow([name_, Path(s_["texture"]).name if s_["texture"] else "", i["face"], i["rot90"], i["confident"],
                    i["reason"], i.get("ratio"), i["auto_face"], i["has_texture"], json.dumps(i.get("scores", {}))])
mem_mb = sum(s_["mesh"].vertices.nbytes + s_["mesh"].faces.nbytes for s_ in SCANS.values()) / 2**20
print(f"{len(SCANS)} scan, {sum(not s_['info']['confident'] for s_ in SCANS.values())} cần xem -> "
      f"{OUT_SOAT / 'audit.csv'} | mesh giữ trong RAM ≈ {mem_mb:.0f} MB (chưa tính texture)")
""")

# =====================================================================================================
# PHẦN A
# =====================================================================================================
md(r"""
---
## 5. Phần A — Render giả ảnh chụp (pyrender, GPU)
`PRESET="mixed"` trộn topdown / handheld / raking / closeup: góc camera, FOV, khoảng cách, xoay trong khung, loại/hướng/
nhiệt độ màu đèn, hậu kỳ camera đều lấy ngẫu nhiên (`sample_shot`). Khung góc thấp (raking) có thể nhìn nghiêng cạnh và
tối — đúng thực tế; muốn nhiều khung thấy rõ mặt khắc thì đổi `PRESET="topdown"` hoặc `"closeup"`.
Mỗi scan có `*_meta.json`: pose camera, đèn, hậu kỳ, vật che, và `face_info` (mặt đã chọn + điểm) để truy lại.
In kèm thời gian render trung bình mỗi ảnh.
""")

code(r"""
if not (RUN_A and EGL_OK):
    print("BỎ QUA phần A:", "RUN_A = False" if not RUN_A else "không có GPU/EGL")
else:
    for name_, s_ in SCANS.items():
        info = s_["info"]
        if not info["confident"] and not RENDER_UNCONFIDENT:
            print(f"BỎ QUA {name_}: chưa chắc mặt khắc ({info['reason']}) — sửa manifest.csv rồi chạy lại")
            continue
        stem = s_["path"].stem
        bw, bh = s_["mesh"].metadata["block_size_mm"][:2]
        t0 = time.time()
        be = PyrenderBackend(s_["mesh"], RENDER_W, RENDER_H)
        recs = render_dataset(be, OUT_A / stem, N_SHOTS, (bw, bh), RENDER_W, RENDER_H, preset=PRESET, seed=3,
                              name=stem, occluder_prob=OCCLUDER_PROB,
                              extra_meta={"source_mesh": name_, "face_info": info})
        be.close()
        dt = time.time() - t0
        print(f"{name_}: {N_SHOTS} ảnh {RENDER_W}×{RENDER_H} trong {dt:.0f}s ({dt / N_SHOTS:.2f} s/ảnh)")
        sheet = contact_sheet([OUT_A / stem / r["file"] for r in recs[:16]], cols=4, thumb_w=420)
        cv2.imwrite(str(OUT_A / f"{stem}_sheet.jpg"), sheet); show(OUT_A / f"{stem}_sheet.jpg", figsize=(18, 14))
""")

md(r"""
### (Tuỳ chọn) Thử vật che 2D
Giả tay/thước/nhãn băng dính/lóa flash/bóng người chụp trên scan đầu tiên. Đây là **ảnh demo**, không phải dữ liệu chính;
muốn áp cho cả bộ thì đặt `OCCLUDER_PROB` > 0 ở cell cấu hình.
""")

code(r"""
demo = next((s_ for s_ in SCANS.values() if s_["info"]["confident"] or RENDER_UNCONFIDENT), None)
if RUN_A and EGL_OK and demo is not None:
    bw, bh = demo["mesh"].metadata["block_size_mm"][:2]
    be = PyrenderBackend(demo["mesh"], RENDER_W, RENDER_H)
    rc = render_dataset(be, OUT_A / "occluder_demo", 8, (bw, bh), RENDER_W, RENDER_H, preset=PRESET, seed=9,
                        name="occ", occluder_prob=0.8)
    be.close()
    sheet = contact_sheet([OUT_A / "occluder_demo" / r["file"] for r in rc], cols=4, thumb_w=420)
    cv2.imwrite(str(OUT_A / "occluder_demo_sheet.jpg"), sheet); show(OUT_A / "occluder_demo_sheet.jpg", figsize=(18, 8))
    print("vật che mỗi ảnh:", [r["occluders"] for r in rc])
else:
    print("bỏ qua (phần A không chạy)")
""")

# =====================================================================================================
# PHẦN B
# =====================================================================================================
md(r"""
---
## 6. Phần B — Tăng cường hình học, độc lập ánh sáng (CPU)

```
mặt khắc hướng +Z ──chiếu trực giao (Z-buffer, rasterize tam giác)──> ảnh độ sâu 2D ──tăng cường──> ảnh kết quả
                     elev/azim/roll tuỳ ý                                (mm)            6 phương pháp
```

Ảnh độ sâu là mặt Monge `z = f(u,v)` trong hệ quy chiếu camera, nên toàn bộ hình học vi phân chạy bằng
numpy/cv2/scipy thay cho tính toán trên mesh hàng triệu mặt.

| # | Phương pháp | Công thức lõi |
|---|---|---|
| 1 | Bản đồ độ sâu | `h - G_σ * h` (khử nền thấp tần) |
| 2 | Bản đồ pháp tuyến | `n = normalize(-hx, -hy, 1)` |
| 3 | Độ cong (Monge) | `H = ((1+q²)r - 2pqs + (1+p²)t) / (2W³)` |
| 4 | MSII | `v_r ≈ ½ + 3/(4r)·(trung_bình_đĩa_r(h) - h)` |
| 5 | Ambient Occlusion | `AO = trung_bình_φ[ 1/(1 + tan²θ_h) ]` |
| 6 | Exaggerated Shading + Radiance Scaling | `S_i = ½ + c·(s_i - s_{i+1})`; `σ(κ̄) = ((α+1)+(α-1)κ̄)/((α+1)-(α-1)κ̄)` |

Phân tích chi tiết chạy trên **một** scan (`SCAN`); cuối phần B xuất ảnh tăng cường cho **mọi** scan. Mesh lấy từ bộ
nhớ (`SCANS`), không nạp lại.

### Chọn scan phân tích
Kiểm bằng mắt: **trái** là mặt được chọn (phải có nét khắc), **phải** là mặt đối diện.
""")

code(r"""
SCAN_NAME = SCAN or next(iter(SCANS))
mesh, fi = SCANS[SCAN_NAME]["mesh"], SCANS[SCAN_NAME]["info"]
print(f"{SCAN_NAME}: {len(mesh.vertices):,} đỉnh, {len(mesh.faces):,} mặt, khối(mm) {np.round(mesh.bounding_box.extents, 1)}")
print(f"mặt khắc {fi['face']} — {fi['reason']}" + ("" if fi["confident"] else "   <<< CHƯA CHẮC: kiểm ảnh bên dưới"))

P_top = project_depth(mesh, 90, 0, px_mm=PX_MM)
PX_MM = P_top["px_mm"]                                # mọi phép chiếu sau dùng CHUNG bước lưới này
P_bot = project_depth(mesh, -90, 0, px_mm=PX_MM)
print(f"px_mm = {PX_MM:.4f} mm/px -> ảnh nhìn thẳng {P_top['depth'].shape}")

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
for a_, P, t_ in zip(ax, (P_top, P_bot), ("mặt được chọn (+Z)", "mặt đối diện (-Z)")):
    loc = depth_map(P["depth"], PX_MM, hp_sigma_px=12.0)["local"]
    s_ = float(np.percentile(np.abs(loc[P["valid"]]), 99)) or 1.0
    a_.imshow(np.where(P["valid"], loc, np.nan), cmap="coolwarm", vmin=-s_, vmax=s_, interpolation="nearest")
    a_.set_title(f"{t_}: độ sâu cục bộ ±{s_:.2f} mm"); a_.axis("off")
plt.tight_layout(); plt.show()
""")

md("### Kiểm chứng giải tích\n"
   "Trước khi tin bất kỳ bức ảnh nào, kiểm các đại lượng có **đáp án đóng**: phép chiếu (mặt sin biết trước độ cao, "
   "mặt phẳng nhìn xiên biết trước độ dốc) và các phép tăng cường (bán cầu, mặt phẳng, chân tường, khai triển "
   "Pottmann, hàm Möbius). Bắt trọn các lỗi đường ống hay gặp nhất (đảo trục, quên chia `px_mm`, nhân bậc 2 không "
   "tổng bằng 0) chỉ trong một lần chạy. Có bài FAIL thì cell dừng (`assert`).")

code(r"""
ok = True
def chk(label, cond, detail=""):
    global ok; ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}  {detail}")

def grid_mesh(zfun, n, step):
    xs = (np.arange(n) - (n - 1) / 2) * step
    X, Y = np.meshgrid(xs, xs)
    Vg = np.stack([X, Y, zfun(X, Y)], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    m = trimesh.Trimesh(Vg, np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]), process=False)
    if (m.face_normals[:, 2] < 0).mean() > 0.5:
        m.invert()
    return m

# 0a) phép chiếu, mặt sin: rasterize lấy độ sâu đúng tại tâm pixel, sai số chỉ còn là sai số NỘI SUY TUYẾN TÍNH trên
#     tam giác lưới bước h, cỡ 2·(|f_xx| + |f_yy|)max·h²/8 (hệ số 2 cho cạnh chéo của tam giác)
zf = lambda x, y: np.sin(2 * np.pi * x / 20) * np.cos(2 * np.pi * y / 15)
gm = grid_mesh(zf, 241, 0.5)
P = project_depth(gm, 90, 0, px_mm=0.5); dd, px = P["depth"], P["px_mm"]
Vg = np.asarray(gm.vertices)
Xc, Yc = np.meshgrid(Vg[:, 0].min() + (np.arange(dd.shape[1]) + 0.5) * px, Vg[:, 1].max() - (np.arange(dd.shape[0]) + 0.5) * px)
zc = zf(Xc, Yc); sl = (slice(5, -5), slice(5, -5))
err = np.abs((dd - dd[sl].mean()) - (zc - zc[sl].mean()))[sl]
bound = 2 * ((2 * np.pi / 20) ** 2 + (2 * np.pi / 15) ** 2) * 0.5 ** 2 / 8
chk("chiếu mặt sin: trung vị sai số < 0.01 mm", np.median(err) < 0.01, f"{np.median(err):.4f} mm")
chk("chiếu mặt sin: sai số lớn nhất < giới hạn nội suy", err.max() < bound, f"{err.max():.4f} < {bound:.4f} mm")

# 0b) phép chiếu, mặt phẳng z=0 nhìn xiên: độ sâu tăng theo cột đúng px/tan(elev), không đổi theo hàng
pl = grid_mesh(lambda x, y: 0 * x, 101, 1.0)
for elev in (60.0, 30.0):
    P = project_depth(pl, elev, 0, px_mm=0.5)
    r_, c_ = np.nonzero(P["valid"])
    kk = np.linalg.lstsq(np.stack([c_, r_, np.ones_like(c_)], 1).astype(float), P["depth"][r_, c_], rcond=None)[0]
    pred = 0.5 / math.tan(math.radians(elev))
    chk(f"chiếu mặt phẳng elev={elev:.0f}°: dốc theo cột = px/tan(elev)", abs(kk[0] - pred) / pred < 0.005,
        f"{kk[0]:.5f} vs {pred:.5f}")
    chk(f"chiếu mặt phẳng elev={elev:.0f}°: không dốc theo hàng", abs(kk[1]) < 1e-3, f"{kk[1]:+.6f}")

# 1) Bán cầu bán kính R:  H = -1/R,  K = +1/R^2  (quy ước Monge: phần NỔI có H < 0)
for px_mm, R_px in [(0.386, 120), (1.0, 120)]:
    N = 2*R_px + 41; cc = N // 2
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float64)
    R_mm = R_px * px_mm
    hs = np.sqrt(np.maximum(R_mm**2 - ((xx-cc)**2 + (yy-cc)**2) * px_mm**2, 0.0)).astype(np.float32)
    cv_ = curvature(hs, px_mm, sigma_px=1.5)
    Hc, Kc = cv_["H"][cc, cc], cv_["K"][cc, cc]
    chk(f"bán cầu px_mm={px_mm}  H=-1/R", abs(Hc + 1/R_mm)/(1/R_mm) < 0.05, f"{Hc:+.5f} vs {-1/R_mm:+.5f}")
    chk(f"bán cầu px_mm={px_mm}  K=+1/R²", abs(Kc - 1/R_mm**2)/(1/R_mm**2) < 0.05, f"{Kc:+.6f} vs {1/R_mm**2:+.6f}")

# 2) Mặt phẳng: AO = 1, MSII v_r = 1/2, H = K = 0
flat = np.zeros((200, 200), np.float32)
chk("mặt phẳng AO = 1", abs(horizon_ao(flat, 0.386, 16, 24)[100,100] - 1) < 1e-5)
chk("mặt phẳng MSII = ½", np.abs(msii_volume(flat, 0.386, (6,12,24))[100,100] - 0.5).max() < 1e-5)
chk("mặt phẳng H = 0", abs(curvature(flat, 0.386)["H"][100,100]) < 1e-6)

# 3) Chân tường đứng: 7/16 phương vị bị che -> AO ≈ 0.5625
wall = np.zeros((200, 200), np.float32); wall[:, 100:] = 50.0
ao_w = horizon_ao(wall, 1.0, 16, 24)
chk("chân tường AO ≈ 0.5", 0.45 < ao_w[100, 99] < 0.62, f"{ao_w[100,99]:.4f}")
chk("xa tường  AO = 1",    abs(ao_w[100, 10] - 1) < 1e-4, f"{ao_w[100,10]:.6f}")

# 4) MSII khớp khai triển Pottmann v_r - ½ = (3/16)·H·r trên vòm cầu
NN, c0, px_, R_ = 401, 200, 0.386, 60.0
y2, x2 = np.mgrid[0:NN, 0:NN].astype(np.float64)
dome = np.sqrt(np.maximum(R_**2 - ((x2-c0)**2 + (y2-c0)**2) * px_**2, 0.0)).astype(np.float32)
Vd = msii_volume(dome, px_, (12, 24))
for j, rpx in enumerate((12, 24)):
    pred = (3/16) * (-1/R_) * (rpx * px_); got = Vd[c0, c0, j] - 0.5
    chk(f"Pottmann r={rpx}px", abs(got-pred)/abs(pred) < 0.10, f"{got:+.5f} vs {pred:+.5f}")

# 5) Radiance scaling: ba tính chất xác định hàm Möbius
for al in (2.0, 3.0, 5.0):
    sig_ = lambda kb: ((al+1) + (al-1)*kb) / ((al+1) - (al-1)*kb)
    chk(f"σ(κ̄) với α={al}", abs(sig_(0)-1) < 1e-12 and abs(sig_(1)-al) < 1e-12 and abs(sig_(-1)-1/al) < 1e-12)

print("\n" + ("TẤT CẢ PASS" if ok else "CÓ TEST FAIL — dừng lại và sửa trước khi đọc ảnh"))
assert ok
""")

md("""### Bước 1 — Chiếu scan từ nhiều góc

Khi `elev` giảm, ảnh **co ngắn** theo phương nghiêng (foreshortening) và tỉ lệ pixel bị che khuất tăng dần — đó
là hình học đúng, không phải lỗi. Mọi góc dùng **chung** bước lưới `PX_MM` để so sánh được.

Chú ý: nét khắc vẫn **đọc được ở mọi góc**, vì tăng cường dựa trên hình học chứ không dựa vào đèn.""")

code(r"""
proj = {}
for elev, azim in ANGLES:
    t0 = time.time(); P = project_depth(mesh, elev_deg=elev, azim_deg=azim, px_mm=PX_MM)
    proj[(elev, azim)] = P
    print(f"elev={elev:3d}° azim={azim:3d}°  ảnh {str(P['depth'].shape):14s} "
          f"hợp lệ {P['valid'].mean()*100:5.1f}%  dải độ sâu {P['depth'].max():7.2f} mm  "
          f"{(time.time()-t0)*1000:5.0f} ms")

fig, ax = plt.subplots(1, len(ANGLES), figsize=(4.3*len(ANGLES), 5.5))
for a_, (key_, P) in zip(ax, proj.items()):
    dep = np.where(P["valid"], P["depth"], np.nan)        # để trống chỗ bị che, không bôi nhoè
    a_.imshow(flip(dep), cmap="viridis", interpolation="nearest")
    a_.set_title(f"elev={key_[0]}° azim={key_[1]}°\nảnh độ sâu đã chiếu"); a_.axis("off")
plt.tight_layout(); plt.show()

# tăng cường (AO) trên chính các ảnh đã chiếu
fig, ax = plt.subplots(1, len(ANGLES), figsize=(4.3*len(ANGLES), 5.5))
for a_, (key_, P) in zip(ax, proj.items()):
    hh = prepare_height(P["depth"], P["px_mm"], SIGMA0_PX)
    ao_ = np.where(P["valid"], horizon_ao(hh, P["px_mm"], 12, AO_R_PX), np.nan)
    a_.imshow(flip(ao_), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a_.set_title(f"elev={key_[0]}° — AO sau khi chiếu"); a_.axis("off")
plt.tight_layout(); plt.show()
""")

md("""### Làm trơn σ₀, đo độ sâu nét khắc, tự chọn vùng zoom

Ảnh độ sâu của scan là mặt ghép từ các **tam giác phẳng** cộng nhiễu đo; đạo hàm bậc hai của nó là dãy xung trên cạnh
tam giác, nên phải làm trơn σ₀ ≈ 1 px trước (xem `prepare_height`).

Vùng zoom (`CROP_MM`) được **tự đặt vào chỗ nhiều nét khắc nhất**: mật độ vách dốc ngắn (độ dốc > 1), sau khi bỏ các
đường thẳng dài (khe nứt, mép khối) bằng phép mở hình thái và bỏ dải mép 5 %; cửa sổ phải nằm trọn trong lõi mặt khắc.
Chọn theo biên độ độ sâu thì cửa sổ sẽ rơi vào khe nứt (chỗ sâu nhất).""")

code(r"""
# Ảnh độ sâu dùng cho toàn bộ phần tăng cường = KẾT QUẢ CHIẾU. Đổi VIEW là mọi phương pháp chạy lại ở góc đó.
VIEW = (90, 0)
P_view = proj[VIEW] if VIEW in proj else project_depth(mesh, *VIEW, px_mm=PX_MM)
h_raw, VALID = P_view["depth"], P_view["valid"]
h = prepare_height(h_raw, PX_MM, sigma0_px=SIGMA0_PX)
loc = depth_map(h, PX_MM, hp_sigma_px=12.0)["local"]
# biên độ nét khắc ĐIỂN HÌNH: p5–p95, không p1–p99 — vài khe sâu (vd. khe nứt xuyên khối, rãnh ghép) chiếm
# chưa tới 1% diện tích nhưng sâu tới đáy khối và sẽ thổi phồng con số này
RELIEF_MM = float(np.subtract(*np.percentile(loc[VALID], [95, 5])))
st = relief_stats(h_raw, PX_MM)
print(f"góc nhìn {VIEW}  ->  ảnh độ sâu {h_raw.shape}")
for key_, v in st.items():
    print(f"  {key_:12s} {v}")
print(f"  relief nét khắc điển hình (p5–p95 sau khử nền) = {RELIEF_MM:.3f} mm ≈ {RELIEF_MM / PX_MM:.1f} px")
print(f"  (p1–p99 = {np.subtract(*np.percentile(loc[VALID], [99, 1])):.3f} mm: lớn hơn nhiều nếu có khe sâu)")

# vùng zoom: cửa sổ CROP_MM quanh chỗ chi tiết DÀY nhất, nằm trọn trong mặt khắc. Chi tiết = mật độ VÁCH DỐC NGẮN
# (> 45°): đo biên độ độ sâu thì một khe sâu thắng; còn cấu trúc thẳng DÀI (khe ghép, mép khối, đường kẻ cột) bị tách
# ra bằng phép mở với phần tử đoạn thẳng ngang / dọc dài CROP/3 — nét chữ, hoa văn là các vách ngắn nên giữ lại.
cpx = max(int(CROP_MM / PX_MM), 32)
dv0 = derivatives(h, PX_MM, 1.0)
steep = ((np.hypot(dv0["hx"], dv0["hy"]) > 1.0) & VALID).astype(np.uint8)
Lz = max(cpx // 3, 9)
lines = cv2.morphologyEx(steep, cv2.MORPH_OPEN, np.ones((1, Lz), np.uint8)) | \
        cv2.morphologyEx(steep, cv2.MORPH_OPEN, np.ones((Lz, 1), np.uint8))
lines = cv2.dilate(lines, np.ones((5, 5), np.uint8))
rim = max(int(0.05 * max(h.shape)), 3) | 1           # bỏ viền khối (gờ mép cũng là vách ngắn nhưng không phải nội dung)
core = cv2.erode(np.pad(VALID, 1).astype(np.uint8), np.ones((rim, rim), np.uint8))[1:-1, 1:-1]
# tâm hợp lệ = cả cửa sổ nằm trong lõi. Đệm False quanh ảnh: cv2.erode mặc định coi ngoài ảnh là hợp lệ.
kw_ = cpx // 2 * 2 + 1
inner = cv2.erode(np.pad(core, kw_), np.ones((kw_, kw_), np.uint8))[kw_:-kw_, kw_:-kw_].astype(bool)
en = cv2.GaussianBlur((steep & (1 - lines) & core).astype(np.float32), (0, 0), cpx / 4)
en[~inner] = 0
cy, cx = np.unravel_index(int(np.argmax(en)), en.shape) if inner.any() else (h.shape[0] // 2, h.shape[1] // 2)
CROP = tuple(int(v) for v in (max(cy - cpx // 2, 0), min(cy + cpx // 2, h.shape[0]),
                              max(cx - cpx // 2, 0), min(cx + cpx // 2, h.shape[1])))
cr = lambda a: flip(a[CROP[0]:CROP[1], CROP[2]:CROP[3]])
print("vùng zoom (y0, y1, x0, x1):", CROP, f"= {cpx * PX_MM:.0f} mm")

# mặt cắt ngang qua tâm vùng zoom: thấy ngay relief, độ rộng mép, và nhiễu trong một hình
seg = slice(CROP[2], CROP[3])
plt.figure(figsize=(14, 3.2))
plt.plot(np.arange(seg.start, seg.stop), h_raw[cy, seg] - np.median(h_raw[cy, seg]), lw=1.0, label="thô")
plt.plot(np.arange(seg.start, seg.stop), h[cy, seg] - np.median(h[cy, seg]), lw=1.8, label=f"sau σ₀={SIGMA0_PX}px")
plt.axhline(0, color="k", lw=0.6)
plt.xlabel("cột (px)"); plt.ylabel("độ cao (mm)")
plt.title(f"Mặt cắt ngang hàng {cy} — px_mm={PX_MM:.4f}")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
show(cr(np.where(VALID, loc, np.nan)), "vùng zoom — độ sâu cục bộ", figsize=(7, 7), cmap="coolwarm")
""")


def method_cells(num, title, explain, body):
    md(f"### Phương pháp {num} — {title}\n{explain}")
    code(body)


method_cells(1, "Bản đồ độ sâu", """
Bản thô có đơn vị vật lý (mm) nhưng bị chi phối bởi dạng tổng thể của khối (cong, vênh, nghiêng) — nét khắc chỉ là
phần nhỏ trong dải màu. Bản `local` (trừ nền thấp tần) mới là bản dùng được, và nó cũng là bản chịu được khối cong/vênh.
""", r"""
dm = depth_map(h, PX_MM, hp_sigma_px=12.0)
print(f"độ cao: [{dm['lo_mm']:.3f}, {dm['hi_mm']:.3f}] mm (percentile 1–99, đã loại viền)")

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
im = ax[0].imshow(cr(dm["mm"]), cmap="viridis", interpolation="nearest")
ax[0].set_title("độ cao thô [mm]"); ax[0].axis("off"); fig.colorbar(im, ax=ax[0], fraction=.046, label="mm")
amp = float(np.percentile(np.abs(dm["local"][VALID]), 99))
ax[1].imshow(cr(dm["local"]), cmap="coolwarm", vmin=-amp, vmax=amp, interpolation="nearest")
ax[1].set_title(f"local = h - G₁₂ₚₓ*h   ±{amp:.3f} mm"); ax[1].axis("off")
plt.tight_layout(); plt.show()

save(flip(colorize(norm01(dm["local"], sym=True), "coolwarm")), "01_depth_local.png")
""")

method_cells(2, "Bản đồ pháp tuyến", """
Mép nét khắc dốc hơn nhiều so với nền nên normal map **thô** thường cho ra nền một màu cộng viền cháy sáng — đúng về
kỹ thuật nhưng khó đọc. Hệ số `gain < 1` nén độ dốc lại. Kèm panel **slope/aspect HSV**: mắt người phân tách nét
theo *hướng dốc*, nên bản này thường đọc rõ hơn normal map RGB.
""", r"""
for g in (1.0, 0.35):
    n = normals(h, PX_MM, sigma_px=1.0, gain=g)
    show(cr(normal_map_rgb(n)), f"normal map, gain={g}" + ("  (thô)" if g == 1.0 else "  (đã nén độ dốc)"),
         figsize=(8, 8))

hsv = slope_aspect_hsv(h, PX_MM, sigma_px=1.0)
show(cr(hsv), "slope / aspect (HSV): hue = hướng dốc", figsize=(8, 8))
save(flip(normal_map_rgb(normals(h, PX_MM, 1.0, 0.35))), "02_normal.png")
save(flip(hsv), "02_slope_aspect.png")
""")

method_cells(3, "Độ cong (Monge patch chính xác)", """
`H ≈ ½∇²h` **chỉ** đúng khi độ dốc ≲ 0.3. Cell dưới đo độ dốc thật của scan và tỉ số lệch của xấp xỉ Laplacian
trên 1% pixel cong nhất. Tỉ số có thể gần 1 ngay cả khi độ dốc rất lớn: pixel cong nhất thường nằm ở đỉnh/chân vách,
nơi độ dốc cục bộ lại nhỏ — công thức chính xác đúng ở mọi nơi nên không phải đoán trường hợp nào.

`K` được hiện cho đủ nhưng **không** vào lưới so sánh: nét khắc gần *developable* (sống + vách) nên `K ≈ 0` gần khắp
nơi và panel K chủ yếu là nhiễu.
""", r"""
cvt = curvature(h, PX_MM, sigma_px=CURV_SIGMA)
dv = derivatives(h, PX_MM, CURV_SIGMA)
lap_H = 0.5 * (dv["hxx"] + dv["hyy"])                      # xấp xỉ Laplacian
edge = VALID & (np.abs(cvt["curvedness"]) > np.percentile(np.abs(cvt["curvedness"][VALID]), 99))
print(f"trên 1% pixel cong nhất: |Laplacian/2| / |H| trung vị = "
      f"{np.median(np.abs(lap_H[edge]) / (np.abs(cvt['H'][edge]) + 1e-9)):.1f}×  <- sai số nếu dùng Laplacian")
print(f"độ dốc p99 = {np.percentile(np.hypot(dv['hx'], dv['hy'])[VALID], 99):.2f}  (xấp xỉ Laplacian cần ≲ 0.3)")

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
aH = float(np.percentile(np.abs(cvt["H"][VALID]), 99))
ax[0].imshow(cr(cvt["H"]), cmap="coolwarm", vmin=-aH, vmax=aH, interpolation="nearest")
ax[0].set_title(f"H [1/mm]  ±{aH:.2f}\n(âm = phần nổi)")
aK = float(np.percentile(np.abs(cvt["K"][VALID]), 99))
ax[1].imshow(cr(cvt["K"]), cmap="coolwarm", vmin=-aK, vmax=aK, interpolation="nearest")
ax[1].set_title(f"K [1/mm²]  ±{aK:.3f}")
ax[2].imshow(cr(cvt["shape_index"] * norm01(cvt["curvedness"])), cmap="coolwarm", vmin=-1, vmax=1,
             interpolation="nearest")
ax[2].set_title("shape index × curvedness\n(điều biên để nền phẳng về trung tính)")
for a_ in ax: a_.axis("off")
plt.tight_layout(); plt.show()

save(flip(METHODS["curvature"](h, PX_MM, **METHOD_KW)["img"]), "03_curvature.png")
""")

method_cells(4, "Bất biến tích phân đa tỉ lệ (MSII)", """
`v_r(p)` = tỉ lệ thể tích khối nằm trong quả cầu bán kính `r` quanh điểm. Mặt phẳng cho đúng `½`; khai triển
Pottmann cho `v_r − ½ = (3/16)·H·r` (quy ước Monge của module này).

Xấp xỉ convolution chỉ hợp lệ khi `r_mm ≳ 2·relief`; bán kính nhỏ hơn thì **bão hoà** và chỉ nên đọc định tính.
Cell dưới dùng biên độ nét khắc đo được trên chính scan (`RELIEF_MM`) và đối chiếu xấp xỉ với cầu phương chính xác.
""", r"""
Vm = msii_volume(h, PX_MM, MSII_RADII)
comb = msii_combine(Vm, MSII_RADII, PX_MM)

print(f"relief nét khắc đo được = {RELIEF_MM:.3f} mm")
print("r(px)   r(mm)   max|v-½| lý thuyết   đo được")
for j, r in enumerate(MSII_RADII):
    r_mm = r * PX_MM
    print(f"{r:5d} {r_mm:7.2f} {3*RELIEF_MM/(4*r_mm):16.2f} {np.abs(Vm[...,j]-0.5)[VALID].max():11.3f}"
          + ("   <- bão hoà" if 3*RELIEF_MM/(4*r_mm) > 0.4 else ""))

# thang màu KHOÁ CHUNG, nếu không mỗi panel tự chuẩn hoá và dãy sweep chẳng cho thấy gì
amp = float(np.percentile(np.abs(Vm[..., len(MSII_RADII)//2] - 0.5)[VALID], 99))
fig, ax = plt.subplots(1, len(MSII_RADII), figsize=(4*len(MSII_RADII), 4.6))
for j, r in enumerate(MSII_RADII):
    ax[j].imshow(cr(Vm[..., j] - 0.5), cmap="coolwarm", vmin=-amp, vmax=amp, interpolation="nearest")
    ax[j].set_title(f"r = {r} px"); ax[j].axis("off")
plt.suptitle(f"v_r − ½  (thang chung ±{amp:.3f})"); plt.tight_layout(); plt.show()

show(cr(comb["scale_argmax"]), "tỉ lệ đặc trưng (argmax |v−½|)", figsize=(7, 7), cmap="turbo")

if 24 in MSII_RADII:
    Ve = msii_volume(h[CROP[0]:CROP[1], CROP[2]:CROP[3]], PX_MM, (24,), exact=True, n_rho=8, n_phi=16)
    Vc = Vm[CROP[0]:CROP[1], CROP[2]:CROP[3], MSII_RADII.index(24)]
    print(f"\nr=24px  xấp xỉ convolution vs cầu phương chính xác: "
          f"lệch trung vị {np.median(np.abs(Ve[...,0]-Vc)):.5f}, max {np.abs(Ve[...,0]-Vc).max():.5f}")
save(flip(METHODS["msii"](h, PX_MM, **METHOD_KW)["img"]), "04_msii.png")
""")

method_cells(5, "Ambient Occlusion (horizon mapping)", """
`AO = trung_bình_φ[ 1/(1+tan²θ_h) ]` (xem `horizon_ao`). Rãnh khắc và vết nứt thường hiện rõ nhất ở đây. Bán kính tìm
chân trời quyết định thang chi tiết được làm nổi — cell dưới quét vài bán kính trên **cùng thang [0,1]** để chọn
`AO_R_PX` cho scan mới.
""", r"""
t0 = time.time(); ao = horizon_ao(h, PX_MM, AO_N_AZ, AO_R_PX); print(f"AO: {time.time()-t0:.2f}s")
print(f"AO ∈ [{ao[VALID].min():.3f}, {ao[VALID].max():.3f}]  (1 = hở hoàn toàn)")

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[0].imshow(flip(np.where(VALID, ao, np.nan)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")  # KHÔNG kéo giãn
ax[0].set_title("AO toàn mặt khắc  [0,1] không kéo giãn"); ax[0].axis("off")
ax[1].imshow(cr(ao), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax[1].set_title("AO — zoom"); ax[1].axis("off")
plt.tight_layout(); plt.show()

ao_radii = (8, AO_R_PX, 3 * AO_R_PX)
fig, ax = plt.subplots(1, len(ao_radii), figsize=(5.5 * len(ao_radii), 6))
for a_, r in zip(ax, ao_radii):
    a_.imshow(cr(horizon_ao(h, PX_MM, 12, r)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a_.set_title(f"AO bán kính {r} px = {r * PX_MM:.1f} mm"); a_.axis("off")
plt.tight_layout(); plt.show()
save(flip(colorize(ao, "gray")), "05_ao.png")
""")

method_cells(6, "Exaggerated Shading & Radiance Scaling", """
**6a — Exaggerated Shading** tô bóng phần **chi tiết** mà mỗi tỉ lệ thêm vào so với tỉ lệ thô hơn, nên gần như không
đổi theo góc đèn. **6b — Radiance Scaling** nhân bóng Lambert với hàm Möbius của độ cong (lồi sáng lên, lõm tối đi).
Chi tiết công thức xem giải thích của `exaggerated_shading` và `radiance_scaling` ở mục 2B.

**Lambert thuần là đối chứng** — hình thứ hai so Lambert và exaggerated ở ba góc đèn 10°, 25°, 60°.
""", r"""
lam = lambert(h, PX_MM, LIGHT)
exg = exaggerated_shading(h, PX_MM, LIGHT, SIGMA0_PX, n_scales=4, gain=0.4)
rad = radiance_scaling(h, PX_MM, LIGHT, alpha=3.0, sigma_px=CURV_SIGMA)
for nm, im_ in [("Lambert (đối chứng)", lam), ("6a Exaggerated", exg), ("6b Radiance scaling", rad)]:
    print(f"{nm:24s} dải [{im_.min():.3f}, {im_.max():.3f}]")

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
for a_, im_, t_ in zip(ax, (lam, exg, rad), ("Lambert (đối chứng)", "6a Exaggerated", "6b Radiance scaling")):
    a_.imshow(cr(im_), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    a_.set_title(t_); a_.axis("off")
plt.tight_layout(); plt.show()

# điểm mấu chốt: ở góc chiếu THẤP Lambert sập, exaggerated shading thì không
fig, ax = plt.subplots(2, 3, figsize=(17, 11))
for j, elev in enumerate((10.0, 25.0, 60.0)):
    Ld = light_dir(elev, 135.0)
    ax[0, j].imshow(cr(lambert(h, PX_MM, Ld)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax[0, j].set_title(f"Lambert — elev {elev:.0f}°")
    ax[1, j].imshow(cr(exaggerated_shading(h, PX_MM, Ld, SIGMA0_PX)), cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest")
    ax[1, j].set_title(f"Exaggerated — elev {elev:.0f}°")
for a_ in ax.ravel(): a_.axis("off")
plt.tight_layout(); plt.show()

save(flip(colorize(exg, "gray")), "06a_exaggerated.png")
save(flip(colorize(rad, "gray")), "06b_radiance.png")
""")

md("### Lưới so sánh sáu phương pháp\n"
   "Lưới toàn mặt khắc là **bối cảnh**: nén cả mặt vào một ô lưới thường mất chi tiết nhỏ. Lưới **zoom** ngay dưới "
   "mới là kết quả thật. Cả hai đều được ghi ra PNG nguyên độ phân giải để phóng to đọc lại.")

code(r"""
t0 = time.time()
g_full = enhance_grid(h, PX_MM, thumb_w=460, flip_mirror=FLIP_MIRROR, **METHOD_KW)
print(f"lưới toàn mặt {g_full.shape}  ({time.time()-t0:.1f}s)")
save(g_full, "grid_full.png"); show(g_full, "Sáu phương pháp — toàn mặt khắc", figsize=(17, 15))
""")

code(r"""
g_crop = enhance_grid(h, PX_MM, crop=CROP, thumb_w=420, flip_mirror=FLIP_MIRROR, **METHOD_KW)
save(g_crop, "grid_crop.png")
show(g_crop, f"Sáu phương pháp — vùng zoom {CROP_MM:.0f} mm  ← hình quan trọng nhất", figsize=(17, 12))
""")

md(r"""
### Xuất ảnh tăng cường cho mọi scan
Chạy `BATCH_METHODS` × `BATCH_ANGLES` cho **mọi scan** (mesh lấy từ bộ nhớ, mặt khắc từ bước soát chung).
Pixel ngoài khối / bị che khuất (`valid = False`) được tô **đen** để không đưa dữ liệu lấp giả vào tập huấn luyện.
Mỗi scan có `batch_meta.json` ghi mặt khắc, góc chiếu, `px_mm`.
""")

code(r"""
BATCH = OUT_B / "batch"; BATCH.mkdir(exist_ok=True)
for name_, s_ in SCANS.items():
    fi_ = s_["info"]
    if not fi_["confident"] and not RENDER_UNCONFIDENT:
        print(f"BỎ QUA {name_}: chưa chắc mặt khắc ({fi_['reason']}) — sửa manifest.csv rồi chạy lại")
        continue
    t0 = time.time()
    m = s_["mesh"]
    px = PX_MM if name_ == SCAN_NAME else project_depth(m, 90, 0)["px_mm"]   # độ phân giải gốc từng scan
    stem = s_["path"].stem
    d_ = BATCH / stem; d_.mkdir(exist_ok=True)
    meta = {"mesh": name_, "face_info": fi_, "px_mm": px, "flip_mirror": FLIP_MIRROR, "images": []}
    for elev, azim in BATCH_ANGLES:
        P = project_depth(m, elev, azim, px_mm=px)
        hh = prepare_height(P["depth"], px, SIGMA0_PX)
        for nm in BATCH_METHODS:
            img = METHODS[nm](hh, px, **METHOD_KW)["img"].copy()
            img[~P["valid"]] = 0
            fn = f"{stem}_e{elev}_a{azim}_{nm}.png"
            cv2.imwrite(str(d_ / fn), cv2.cvtColor(flip(img), cv2.COLOR_RGB2BGR))
            meta["images"].append({"file": fn, "elev": elev, "azim": azim, "method": nm,
                                   "valid_frac": float(P["valid"].mean())})
    (d_ / "batch_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name_}: {len(meta['images'])} ảnh ({time.time() - t0:.0f}s)")
""")

# =====================================================================================================
# Đóng gói
# =====================================================================================================
md("---\n## 7. Đóng gói kết quả\n`outputs.zip` gồm `soat/`, `A_render/` (nếu phần A chạy) và `B_enhance/`.")

code(r"""
shutil.make_archive("/kaggle/working/outputs", "zip", OUT)
for sub in (OUT_SOAT, OUT_A, OUT_B):
    files = [q for q in sub.rglob("*") if q.is_file()]
    print(f"{sub.name:10s} {len(files):5d} file  {sum(q.stat().st_size for q in files) // 1024:8d} KB")
print(f"zip: {Path('/kaggle/working/outputs.zip').stat().st_size // 1024} KB")
""")

# =====================================================================================================
# Kiểm tra trước khi ghi
# =====================================================================================================
# 1) cell thư viện ghép lại = đúng cây cú pháp của 2 module gốc
got = ast.dump(ast.Module(body=_body_for_compare("\n\n".join(LIB_CODE)), type_ignores=[]))
want = ast.dump(ast.Module(body=_body_for_compare(render_src) + _body_for_compare(enhance_src), type_ignores=[]))
if got != want:
    raise SystemExit("cell thư viện KHÔNG khớp mã nguồn module gốc (AST khác)")

# 2) biến của các cell chạy không được trùng tên hàm / hằng số thư viện
lib_set = set(lib_names)
clash = set()
for s in EXEC_CODE:
    try:
        t = ast.parse("\n".join(l for l in s.splitlines() if not l.lstrip().startswith("!")))
    except SyntaxError as e:
        raise SystemExit(f"cell chạy lỗi cú pháp: {e}\n{s[:300]}")
    for n in t.body:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store) and x.id in lib_set:
                clash.add(x.id)
            elif isinstance(x, (ast.FunctionDef, ast.ClassDef)) and x.name in lib_set:
                clash.add(x.name)
if clash:
    raise SystemExit(f"cell chạy ghi đè tên của thư viện: {sorted(clash)}")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = HERE / "mocban_scan3d_kaggle.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
n_code = sum(c["cell_type"] == "code" for c in cells)
print("đã ghi", out, f"({len(cells)} cell: {n_code} code, {len(cells) - n_code} markdown; "
      f"{len(LIB_CODE)} cell thư viện, khớp AST module gốc; không trùng tên)")

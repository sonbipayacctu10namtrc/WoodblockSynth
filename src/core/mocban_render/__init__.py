"""
mocban_render — package; tên file sNN_ = thứ tự pipeline = thứ tự trong notebook.
  s01_scan.py          Nạp mesh scan, tìm texture rời, đưa scan về khung OBB chuẩn (prepare_scan).
  s02_face.py          Tự xác định mặt khắc (detect_main_face), xoay mặt khắc lên +Z (orient_to_face), đọc manifest.
  s03_shot.py          Mô tả một lần chụp (CameraSpec, LightSpec, ShotSpec) và lấy mẫu ngẫu nhiên theo 4 kiểu chụp.
  s04_camera.py        Toán học camera: góc -> hướng, ma trận pose kiểu OpenGL.
  s05_renderer.py      Renderer pyrender: dựng cảnh, render ảnh màu + độ sâu có bóng đổ.
  s06_post.py          Hậu kỳ giả camera (apply_post) và vật che 2D (add_occluders).
  s07_dataset.py       Vòng lặp sinh dữ liệu, soát mặt khắc, ghép ảnh xem nhanh.
"""
from .s01_scan import (  # noqa: F401
    MESH_EXT,
    _IMG_EXT,
    _NOT_ALBEDO,
    _has_texture_image,
    _texture_search_dirs,
    find_texture,
    resolve_texture,
    _proper_rotation,
    prepare_scan,
)
from .s02_face import (  # noqa: F401
    FACE_THIN_RATIO,
    FACE_MIN_RATIO,
    FACE_MIN_COVERAGE,
    FACE_MISSING,
    _axis_names,
    _face_axis,
    _face_height,
    _face_scores,
    detect_main_face,
    orient_to_face,
    load_external_mesh,
    read_manifest,
)
from .s03_shot import (  # noqa: F401
    CameraSpec,
    LightSpec,
    ShotSpec,
    _kelvin_rgb,
    sample_shot,
    frontal_shot,
)
from .s04_camera import (  # noqa: F401
    _spherical_dir,
    look_at_pose,
    camera_pose_from_spec,
)
from .s05_renderer import (  # noqa: F401
    PyrenderBackend,
)
from .s06_post import (  # noqa: F401
    apply_post,
    add_occluders,
)
from .s07_dataset import (  # noqa: F401
    render_dataset,
    face_audit_image,
    audit_scan,
    audit_prepared,
    contact_sheet,
)

FILES = ['s01_scan.py', 's02_face.py', 's03_shot.py', 's04_camera.py', 's05_renderer.py', 's06_post.py', 's07_dataset.py']

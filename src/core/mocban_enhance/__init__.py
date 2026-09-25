"""
mocban_enhance — package; tên file sNN_ = thứ tự pipeline = thứ tự trong notebook.
  s01_projection.py    Chiếu mesh 3D sang ảnh độ sâu 2D (trực giao, góc bất kỳ, rasterize tam giác).
  s02_height.py        Chuẩn bị trường độ cao (làm trơn) và thống kê độ sâu nét.
  s03_derivatives.py   Đạo hàm Gauss (kernel ép đúng trên đa thức) và pháp tuyến.
  s04_depth.py         Phương pháp 1: bản đồ độ sâu cục bộ.
  s05_curvature.py     Phương pháp 3: độ cong Monge chính xác.
  s06_msii.py          Phương pháp 4: bất biến tích phân đa tỉ lệ (MSII).
  s07_ao.py            Phương pháp 5: ambient occlusion (horizon mapping).
  s08_shading.py       Phương pháp 6: exaggerated shading, radiance scaling, Lambert đối chứng.
  s09_display.py       Chuẩn hoá hiển thị: đưa mọi kết quả về ảnh theo một chính sách duy nhất.
  s10_grid.py          Bảng phương pháp METHODS (m_*) và lưới so sánh enhance_grid.
"""
from .s01_projection import (  # noqa: F401
    _spherical_dir,
    view_basis,
    _raster_zbuffer,
    project_depth,
)
from .s02_height import (  # noqa: F401
    prepare_height,
    relief_stats,
)
from .s03_derivatives import (  # noqa: F401
    _dkernel,
    _sepfilt,
    derivatives,
    normals,
    normal_map_rgb,
    slope_aspect_hsv,
)
from .s04_depth import (  # noqa: F401
    depth_map,
)
from .s05_curvature import (  # noqa: F401
    curvature,
    normal_curvature_dir,
)
from .s06_msii import (  # noqa: F401
    _disk_kernel,
    msii_volume,
    msii_combine,
)
from .s07_ao import (  # noqa: F401
    _base_maps,
    _shift,
    horizon_tan,
    horizon_ao,
)
from .s08_shading import (  # noqa: F401
    light_dir,
    lambert,
    exaggerated_shading,
    radiance_scaling,
)
from .s09_display import (  # noqa: F401
    _interior,
    _interior_percentile,
    norm01,
    colorize,
    _lab,
)
from .s10_grid import (  # noqa: F401
    m_depth,
    m_normal,
    m_curvature,
    m_msii,
    m_ao,
    m_exaggerated,
    m_radiance,
    m_lambert,
    METHODS,
    GRID6,
    enhance_grid,
)

FILES = ['s01_projection.py', 's02_height.py', 's03_derivatives.py', 's04_depth.py', 's05_curvature.py', 's06_msii.py', 's07_ao.py', 's08_shading.py', 's09_display.py', 's10_grid.py']

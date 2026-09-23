# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab-only MJCF → USD conversion at the scene construction boundary.

Humanoid articulations are authored as MJCF. IsaacLab consumes USD, so this
module invokes IsaacLab 3 ``MjcfConverter`` / ``MjcfConverterCfg`` lazily and
caches results by absolute MJCF path plus conversion options.

Kit is not required to import this module or to exercise path/config helpers.
Real conversion imports IsaacLab converters only inside
``default_mjcf_converter_factory``.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Union

from filelock import FileLock

from protomotions.robot_configs.base import RobotAssetConfig

# Process-local cache: avoids reconverting when multiple SceneCfg builds share
# the same MJCF + options within one IsaacLab process.
_CONVERSION_CACHE: Dict[Tuple[Any, ...], str] = {}

ConverterFactory = Callable[..., str]
_MJCF_CONVERTER_CACHE_VERSION = "isaaclab3-d6-workaround-v3-textures"
_DEFAULT_ASSET_ROOT = "protomotions/data/assets"


def _resolve_asset_root(asset_root: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve a configured asset root without requiring protomotions.assets."""
    if asset_root and str(asset_root) != _DEFAULT_ASSET_ROOT:
        return Path(asset_root)
    configured_root = os.environ.get("PROTOMOTIONS_ASSET_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    # isaaclab/utils/mjcf_to_usd.py -> protomotions/data/assets
    return Path(__file__).resolve().parents[3] / "data" / "assets"


def _absolute_path(path: Union[os.PathLike, str]) -> str:
    """Expand user paths before resolving them against the current directory."""
    return str(Path(path).expanduser().resolve())


def resolve_robot_mjcf_path(asset: RobotAssetConfig) -> str:
    """Return the absolute MJCF path for a robot asset config."""
    if not asset.asset_file_name:
        raise ValueError("RobotAssetConfig.asset_file_name must be set to an MJCF path")
    asset_file_name = Path(asset.asset_file_name).expanduser()
    if not asset_file_name.is_absolute():
        asset_file_name = _resolve_asset_root(asset.asset_root) / asset_file_name
    return _absolute_path(asset_file_name)


def predicted_converted_usd_path(mjcf_path: str, usd_dir: str) -> str:
    """Return the USD path IsaacLab 3 ``MjcfConverter`` will produce.

    The converter rewrites ``usd_file_name`` to ``{stem}/{stem}.usda`` under
    ``usd_dir``.
    """
    stem = Path(_absolute_path(mjcf_path)).stem
    return _absolute_path(Path(usd_dir).expanduser() / stem / f"{stem}.usda")


def default_usd_cache_dir(mjcf_path: str, options: Mapping[str, Any]) -> str:
    """Stable on-disk cache directory for a MJCF path and conversion options."""
    abs_mjcf = _absolute_path(mjcf_path)
    digest = hashlib.sha1(
        repr(
            (
                _MJCF_CONVERTER_CACHE_VERSION,
                abs_mjcf,
                _mjcf_fingerprint(abs_mjcf),
                _normalized_options(options),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    stem = Path(abs_mjcf).stem
    cache_root = Path("~/.cache/protomotions/isaaclab_mjcf_usd").expanduser()
    return _absolute_path(cache_root / f"{stem}_{digest}")


def build_mjcf_converter_cfg_kwargs(
    mjcf_path: str,
    *,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    self_collision: bool = False,
    fix_base: bool = False,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for IsaacLab 3 ``MjcfConverterCfg`` without importing Kit.

    Only fields present on the public IsaacLab 3 develop API are included.
    Obsolete IsaacLab 2 fields (``import_sites``, ``make_instanceable``) are
    intentionally omitted.
    """
    abs_mjcf = _absolute_path(mjcf_path)
    options = {
        "self_collision": bool(self_collision),
        "fix_base": bool(fix_base),
        "merge_mesh": bool(merge_mesh),
        "collision_from_visuals": bool(collision_from_visuals),
    }
    resolved_usd_dir = (
        _absolute_path(usd_dir)
        if usd_dir is not None
        else default_usd_cache_dir(abs_mjcf, options)
    )
    return {
        "asset_path": abs_mjcf,
        "usd_dir": resolved_usd_dir,
        "force_usd_conversion": bool(force_usd_conversion),
        **options,
    }


def conversion_cache_key(
    cfg_kwargs: Mapping[str, Any],
    *,
    include_source_fingerprint: bool = True,
) -> Tuple[Any, ...]:
    """Stable in-memory cache key for converter cfg kwargs."""
    return (
        _MJCF_CONVERTER_CACHE_VERSION,
        _absolute_path(str(cfg_kwargs["asset_path"])),
        _absolute_path(str(cfg_kwargs["usd_dir"])),
        bool(cfg_kwargs.get("force_usd_conversion", False)),
        bool(cfg_kwargs.get("self_collision", False)),
        bool(cfg_kwargs.get("fix_base", False)),
        bool(cfg_kwargs.get("merge_mesh", False)),
        bool(cfg_kwargs.get("collision_from_visuals", False)),
        (
            _mjcf_fingerprint(str(cfg_kwargs["asset_path"]))
            if include_source_fingerprint
            else None
        ),
    )


def _mjcf_fingerprint(path: str) -> Optional[str]:
    """Hash an MJCF and its referenced assets for cache invalidation.

    IsaacLab's converter consumes mesh and texture files referenced by the
    MJCF, so hashing only the XML timestamp can leave a stale USD after an
    in-place mesh edit. Missing references are included in the digest too, so
    adding a previously missing asset invalidates the cache.
    """
    root_path = Path(_absolute_path(path))
    if not root_path.is_file():
        return None

    digest = hashlib.sha1()
    visited: set[Path] = set()

    def hash_file(file_path: Path) -> None:
        file_path = Path(_absolute_path(file_path))
        digest.update(b"\x00path:")
        digest.update(str(file_path).encode("utf-8"))
        if not file_path.is_file():
            digest.update(b"\x00missing")
            return
        try:
            data = file_path.read_bytes()
        except OSError:
            digest.update(b"\x00unreadable")
            return
        digest.update(b"\x00content:")
        digest.update(data)

    def visit_xml(xml_path: Path) -> None:
        xml_path = Path(_absolute_path(xml_path))
        if xml_path in visited:
            return
        visited.add(xml_path)
        try:
            data = xml_path.read_bytes()
        except OSError:
            hash_file(xml_path)
            return
        digest.update(b"\x00xml:")
        digest.update(str(xml_path).encode("utf-8"))
        digest.update(data)
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return

        compiler = root.find("compiler")
        mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
        texture_dir = compiler.get("texturedir", "") if compiler is not None else ""
        for element in root.iter():
            reference = element.get("file")
            if not reference:
                continue
            if element.tag == "include":
                visit_xml(xml_path.parent / Path(reference).expanduser())
                continue
            directory = mesh_dir if element.tag == "mesh" else texture_dir
            reference_path = Path(reference).expanduser()
            if directory and not reference_path.is_absolute():
                reference_path = Path(directory).expanduser() / reference_path
            hash_file(xml_path.parent / reference_path)

    visit_xml(root_path)
    return digest.hexdigest()


def _normalized_options(options: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((str(k), options[k]) for k in options))


def _conversion_coordination_paths(
    usd_dir: str, cache_key: Tuple[Any, ...]
) -> Tuple[Path, Path]:
    output_dir = Path(_absolute_path(usd_dir))
    key_digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    lock_path = output_dir.with_name(f".{output_dir.name}.protomotions.lock")
    marker_path = output_dir.with_name(
        f".{output_dir.name}.{key_digest}.protomotions-complete"
    )
    return lock_path, marker_path


def _completed_conversion(marker_path: Path) -> Optional[str]:
    try:
        usd_path = _absolute_path(marker_path.read_text().strip())
    except (OSError, ValueError):
        return None
    return usd_path if Path(usd_path).is_file() else None


def _publish_completed_conversion(marker_path: Path, usd_path: str) -> None:
    temporary_marker = marker_path.with_name(
        f"{marker_path.name}.{os.getpid()}.tmp"
    )
    temporary_marker.write_text(usd_path)
    os.replace(temporary_marker, marker_path)


def default_mjcf_converter_factory(**cfg_kwargs: Any) -> str:
    """Run IsaacLab ``MjcfConverter`` and return the generated USD path.

    Requires an active Isaac Sim / Kit runtime.
    """
    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    for extension_id in (
        "isaacsim.asset.importer.utils",
        "isaacsim.asset.importer.mjcf",
    ):
        extension_manager.set_extension_enabled_immediate(extension_id, True)

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from protomotions.simulator.isaaclab.utils.mjcf_d6_workaround import (
        install_isaaclab_mjcf_d6_workaround,
    )

    install_isaaclab_mjcf_d6_workaround()

    converter = MjcfConverter(MjcfConverterCfg(**cfg_kwargs))
    return _absolute_path(converter.usd_path)


def dry_run_mjcf_converter_factory(**cfg_kwargs: Any) -> str:
    """Deterministic Kit-free factory used by unit tests and dry-run mode."""
    return predicted_converted_usd_path(
        cfg_kwargs["asset_path"], cfg_kwargs["usd_dir"]
    )


def convert_mjcf_to_usd(
    mjcf_path: str,
    *,
    converter_factory: Optional[ConverterFactory] = None,
    cache: Optional[MutableMapping[Tuple[Any, ...], str]] = None,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    self_collision: bool = False,
    fix_base: bool = False,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> str:
    """Convert MJCF to USD via IsaacLab 3 APIs, with process-local caching.

    Args:
        mjcf_path: Path to the MJCF file.
        converter_factory: Optional injectable factory ``(**cfg_kwargs) -> usd_path``.
            Defaults to ``default_mjcf_converter_factory``, or
            ``dry_run_mjcf_converter_factory`` when
            ``PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN`` is set.
        cache: Optional cache mapping; defaults to the module-level cache.
        usd_dir: Optional output directory; defaults to a stable user cache path.
        force_usd_conversion: Forwarded to ``MjcfConverterCfg``.
        self_collision: Forwarded to ``MjcfConverterCfg``.
        fix_base: Forwarded to ``MjcfConverterCfg``.
        merge_mesh: Forwarded to ``MjcfConverterCfg``.
        collision_from_visuals: Forwarded to ``MjcfConverterCfg``.

    Returns:
        Absolute path to the generated (or predicted) USD file.
    """
    uses_default_cache = usd_dir is None
    cfg_kwargs = build_mjcf_converter_cfg_kwargs(
        mjcf_path,
        usd_dir=usd_dir,
        force_usd_conversion=force_usd_conversion,
        self_collision=self_collision,
        fix_base=fix_base,
        merge_mesh=merge_mesh,
        collision_from_visuals=collision_from_visuals,
    )
    key = conversion_cache_key(
        cfg_kwargs,
        include_source_fingerprint=not uses_default_cache,
    )
    cache_store = _CONVERSION_CACHE if cache is None else cache
    if key in cache_store and not force_usd_conversion:
        return cache_store[key]

    lock_path, marker_path = _conversion_coordination_paths(
        cfg_kwargs["usd_dir"], key
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if key in cache_store and not force_usd_conversion:
            return cache_store[key]

        if not force_usd_conversion:
            completed_path = _completed_conversion(marker_path)
            if completed_path is not None:
                # Idempotent: repairs caches written before sanitize gained
                # new steps (e.g. contact-report baking) without reconverting.
                if Path(completed_path).is_file():
                    sanitize_converted_mjcf_usd(completed_path)
                cache_store[key] = completed_path
                return completed_path

        if converter_factory is None:
            if os.environ.get("PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN"):
                converter_factory = dry_run_mjcf_converter_factory
            else:
                converter_factory = default_mjcf_converter_factory

        factory_kwargs = cfg_kwargs
        if not force_usd_conversion:
            existing_usd_path = Path(
                predicted_converted_usd_path(
                    cfg_kwargs["asset_path"], cfg_kwargs["usd_dir"]
                )
            )
            if existing_usd_path.is_file():
                # IsaacLab's .asset_hash does not include this workaround's
                # version, so an explicit output directory can otherwise
                # reuse a USD generated before the repair was installed.
                factory_kwargs = {**cfg_kwargs, "force_usd_conversion": True}

        usd_path = _absolute_path(converter_factory(**factory_kwargs))
        if Path(usd_path).is_file():
            sanitize_converted_mjcf_usd(usd_path)
            bind_mjcf_diffuse_textures(usd_path, cfg_kwargs["asset_path"])
        cache_store[key] = usd_path
        if Path(usd_path).is_file():
            _publish_completed_conversion(marker_path, usd_path)
        return usd_path


def sanitize_converted_mjcf_usd(usd_path: str) -> None:
    """Strip MJCF worldbody leftovers the Lab 3 converter bakes into the USD.

    The importer keeps the MJCF ``<geom name="floor">`` and the free-joint
    body's world pose (Atlas Hip at z≈0.96). Isaac Lab then applies
    ``init_state.pos`` / written root state on top, so the robot hovers about
    a meter above the scene ground. Deactivate world floors and zero the
    articulation-root translation so Hip is the asset origin.

    Also bakes ``PhysxContactReportAPI`` onto EVERY rigid body. The converter
    nests bodies in the MJCF kinematic tree, and Isaac Lab's spawn-time
    ``activate_contact_sensors`` stops descending at the first rigid body it
    finds ("nested rigid bodies are not allowed by SDK") -- so only the root
    body gets the API and any ContactSensor on a descendant body dies with
    "could not find any bodies with contact reporter API". Flat pre-built
    USDs get the API on all bodies; this restores that parity.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if prim.GetTypeName() == "Plane" and name in {"floor", "ground"}:
            prim.SetActive(False)
            # Also stamp the defining spec so a Physics variant payload
            # cannot revive the collider.
            spec = prim.GetPrimStack()
            if spec:
                spec[0].active = False
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            # MJCF alpha-0 collision geoms arrive as displayOpacity=[0] but
            # render opaque anyway (spawn-time preview material ignores
            # opacity). Honor the MJCF by marking them purpose='guide':
            # hidden by default, toggleable in the viewport via
            # Show By Purpose -> Guides.
            gprim = UsdGeom.Gprim(prim)
            if gprim:
                opacity = gprim.GetDisplayOpacityAttr().Get()
                if opacity is not None and len(opacity) and not any(opacity):
                    imageable = UsdGeom.Imageable(prim)
                    imageable.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
                    # Undo any earlier visibility-based hiding so the guide
                    # toggle can actually show them.
                    vis = imageable.GetVisibilityAttr()
                    if vis.Get() == UsdGeom.Tokens.invisible:
                        vis.Set(UsdGeom.Tokens.inherited)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # Same recipe as isaaclab.sim.schemas.activate_contact_sensors,
            # minus its early-stop traversal.
            applied = prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in applied:
                prim.AddAppliedSchema("PhysxRigidBodyAPI")
            prim.CreateAttribute(
                "physxRigidBody:sleepThreshold", Sdf.ValueTypeNames.Float
            ).Set(0.0)
            if "PhysxContactReportAPI" not in applied:
                prim.AddAppliedSchema("PhysxContactReportAPI")
            prim.CreateAttribute(
                "physxContactReport:threshold", Sdf.ValueTypeNames.Float
            ).Set(0.0)
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    for layer in stage.GetUsedLayers():
        if not layer.anonymous and layer.dirty:
            layer.Save()


def convert_robot_mjcf_to_usd(
    asset: RobotAssetConfig,
    *,
    converter_factory: Optional[ConverterFactory] = None,
    cache: Optional[MutableMapping[Tuple[Any, ...], str]] = None,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    fix_base: Optional[bool] = None,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> str:
    """Convert a robot asset's MJCF to USD for IsaacLab spawning."""
    return convert_mjcf_to_usd(
        resolve_robot_mjcf_path(asset),
        converter_factory=converter_factory,
        cache=cache,
        usd_dir=usd_dir,
        force_usd_conversion=force_usd_conversion,
        self_collision=bool(asset.self_collisions),
        fix_base=bool(asset.fix_base_link) if fix_base is None else fix_base,
        merge_mesh=merge_mesh,
        collision_from_visuals=collision_from_visuals,
    )


# Per-family PBR values and exceptions carried over from
# usd_convert/patch_atlas_usd_bindings.py (hand-tuned for the Atlas2025 rig
# under Omniverse renderers). Families not listed use the painted-metal
# default. Dark families (Plastic's diffuse averages 0.1) must be dielectric
# (metallic 0) or they render as environment-gray.
_FAMILY_PBR = {
    "Plastic": (0.0, 0.56),
    "Emission": (0.0, 0.5),
    "Bbody": (0.0, 0.48),
}
_FAMILY_PBR_DEFAULT = (0.4, 0.5)
# Families rendered as a plain color constant instead of their texture.
# Emission's shipped texture is a flat green (std ~0.01) — nothing is lost,
# and a constant stays editable in the GUI.
_FAMILY_CONSTANT_COLOR = {
    "Emission": (0.341, 0.906, 0.349),
}


def _parse_mjcf_material_textures(mjcf_path: str) -> Dict[str, Path]:
    """Map MJCF material names to their (existing) diffuse texture files.

    Follows the same path convention as ``_mjcf_fingerprint``: texture file
    references resolve against ``<compiler texturedir>`` relative to the MJCF
    directory. Materials whose texture file is missing are omitted.
    """
    root_path = Path(_absolute_path(mjcf_path))
    try:
        root = ET.fromstring(root_path.read_bytes())
    except (OSError, ET.ParseError):
        return {}
    compiler = root.find("compiler")
    texture_dir = compiler.get("texturedir", "") if compiler is not None else ""
    textures: Dict[str, Path] = {}
    for element in root.iter("texture"):
        name, reference = element.get("name"), element.get("file")
        if not name or not reference:
            continue
        reference_path = Path(reference).expanduser()
        if texture_dir and not reference_path.is_absolute():
            reference_path = Path(texture_dir).expanduser() / reference_path
        textures[name] = Path(
            _absolute_path(root_path.parent / reference_path)
        )
    family_textures: Dict[str, Path] = {}
    for element in root.iter("material"):
        name, texture_name = element.get("name"), element.get("texture")
        if name and texture_name in textures and textures[texture_name].is_file():
            family_textures[name] = textures[texture_name]
    return family_textures


def bind_mjcf_diffuse_textures(usd_path: str, mjcf_path: str) -> None:
    """Wire the MJCF's diffuse textures into the converted USD's materials.

    The Isaac Sim 6 MuJoCo converter binds visual meshes to per-instance
    materials that reference family templates in ``payloads/materials.usda``,
    but (a) drops every MJCF texture, authoring plain white
    ``UsdPreviewSurface`` templates, and (b) writes only ONE family template,
    leaving the other instances' references dangling — the whole robot
    renders untextured. This pass rebuilds the family templates from the
    MJCF's ``<texture>``/``<material>`` tables: missing templates are cloned
    from the authored one, texture images are copied under ``Textures/`` next
    to the root layer, and each family's ``diffuseColor`` is driven by a
    ``UsdUVTexture`` (visual meshes carry ``primvars:st``).

    Purely cosmetic; failures are reported but never abort the conversion.
    """
    try:
        _bind_mjcf_diffuse_textures(usd_path, mjcf_path)
    except Exception as error:  # pragma: no cover - defensive, visual-only
        print(
            f"[WARN] bind_mjcf_diffuse_textures: leaving {usd_path} "
            f"untextured ({type(error).__name__}: {error})"
        )


def _bind_mjcf_diffuse_textures(usd_path: str, mjcf_path: str) -> None:
    import shutil

    from pxr import Gf, Sdf, Usd, UsdShade

    family_textures = _parse_mjcf_material_textures(mjcf_path)
    if not family_textures:
        return
    usd_root = Path(usd_path).parent
    materials_path = usd_root / "payloads" / "materials.usda"
    if not materials_path.is_file():
        return

    stage = Usd.Stage.Open(str(materials_path))
    if stage is None:
        return
    scope = stage.GetPrimAtPath("/Materials")
    if not scope:
        return
    template = next(
        (child for child in scope.GetChildren() if child.IsA(UsdShade.Material)),
        None,
    )
    if template is None or not template.GetChild("PreviewSurface"):
        return

    textures_dir = usd_root / "Textures"
    textures_dir.mkdir(exist_ok=True)
    layer = stage.GetRootLayer()

    for family, texture_path in sorted(family_textures.items()):
        material_path = Sdf.Path(f"/Materials/{family}")
        if not stage.GetPrimAtPath(material_path):
            Sdf.CopySpec(layer, template.GetPath(), layer, material_path)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        surface = UsdShade.Shader(
            stage.GetPrimAtPath(material_path.AppendChild("PreviewSurface"))
        )
        if not material or not surface:
            continue

        # Clones keep the template's absolute connection paths; repoint the
        # material outputs and shader inputs at this family's own prims.
        for output_name in ("surface", "displacement"):
            material.CreateOutput(
                output_name, Sdf.ValueTypeNames.Token
            ).ConnectToSource(
                surface.CreateOutput(output_name, Sdf.ValueTypeNames.Token)
            )
        for input_name, value_type in (
            ("metallic", Sdf.ValueTypeNames.Float),
            ("roughness", Sdf.ValueTypeNames.Float),
            ("opacity", Sdf.ValueTypeNames.Float),
            ("diffuseColor", Sdf.ValueTypeNames.Color3f),
        ):
            surface.CreateInput(input_name, value_type).GetAttr().ClearConnections()

        metallic, roughness = _FAMILY_PBR.get(family, _FAMILY_PBR_DEFAULT)
        surface.GetInput("metallic").Set(metallic)
        surface.GetInput("roughness").Set(roughness)
        surface.GetInput("opacity").Set(1.0)

        constant = _FAMILY_CONSTANT_COLOR.get(family)
        if constant is not None:
            surface.GetInput("diffuseColor").Set(Gf.Vec3f(*constant))
            continue

        local_texture = textures_dir / texture_path.name
        if not local_texture.is_file():
            shutil.copy2(texture_path, local_texture)
        reader = UsdShade.Shader.Define(
            stage, material_path.AppendChild("stReader")
        )
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(
            stage, material_path.AppendChild("diffuseTex")
        )
        texture.CreateIdAttr("UsdUVTexture")
        # Layer-relative: materials.usda sits in payloads/, images one level up.
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            f"../Textures/{local_texture.name}"
        )
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB"
        )
        for wrap in ("wrapS", "wrapT"):
            texture.CreateInput(wrap, Sdf.ValueTypeNames.Token).Set("repeat")
        surface.GetInput("diffuseColor").ConnectToSource(
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )

    layer.Save()


def clear_mjcf_usd_conversion_cache() -> None:
    """Clear the process-local conversion cache (tests only)."""
    _CONVERSION_CACHE.clear()

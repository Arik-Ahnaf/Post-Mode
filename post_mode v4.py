bl_info = {
    "name": "Post Mode",
    "author": "You",
    "version": (0, 4, 0),
    "blender": (5, 2, 0),
    "location": "3D Viewport > Sidebar (N) > Post",
    "description": "Turns a 3D viewport into a top-down wireframe layout view for placing lights and cameras",
    "category": "3D View",
}

import json
import math
import os

import bpy
import gpu
from bpy.app.handlers import persistent
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Quaternion, Vector

# ----------------------------------------------------------------------------
# Settings (edit these to taste). Colors are gamma-space RGB, 0..1.
# ----------------------------------------------------------------------------
CAMERA_COLOR = (1.0, 0.0, 0.0)     # red
SELECT_COLOR = (0.0, 0.1, 0.75)    # deep blue

# Light colors by light type (drawn by the add-on on top of Blender's own light display)
LIGHT_COLORS = {
    'SPOT': (0.1, 0.9, 0.1),       # green
    'AREA': (1.0, 0.0, 1.0),       # magenta
    'POINT': (0.0, 0.9, 1.0),      # cyan
    'SUN': (1.0, 0.5, 0.0),        # orange
}

# Object types that stay visible in the Post viewport (everything else is hidden)
VISIBLE_TYPES = {"camera", "light"}

# Theme properties we override: ThemeView3D property name -> color
THEME_OVERRIDES = {
    "camera": CAMERA_COLOR,            # unselected cameras
    "object_selected": SELECT_COLOR,   # selected
    "object_active": SELECT_COLOR,     # active selected
}

# Module state: None when Post mode is off, otherwise a dict of saved settings
_saved = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _find_area(area_ptr):
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.as_pointer() == area_ptr:
                return win, area
    return None, None


def _window_region(area):
    return next((r for r in area.regions if r.type == 'WINDOW'), None)


def _view3d_space(area):
    for sp in area.spaces:
        if sp.type == 'VIEW_3D':
            return sp
    return None


def _theme():
    return bpy.context.preferences.themes[0].view_3d


def _color_props(theme):
    return [
        p.identifier
        for p in theme.bl_rna.properties
        if p.type == 'FLOAT' and p.subtype in {'COLOR', 'COLOR_GAMMA'} and p.array_length > 0
        and p.identifier != "rna_type"
    ]


def _backup_path():
    return os.path.join(bpy.utils.user_resource('CONFIG', create=True), "post_mode_theme_backup.json")


def _snapshot_theme():
    """Save the whole 3D Viewport color palette in memory AND on disk (crash/reload safety)."""
    theme = _theme()
    snap = {}
    for key in _color_props(theme):
        try:
            snap[key] = list(getattr(theme, key))
        except (AttributeError, TypeError):
            pass
    try:
        with open(_backup_path(), "w") as f:
            json.dump(snap, f)
    except OSError:
        pass
    return snap


def _apply_theme_overrides():
    theme = _theme()
    for key, color in THEME_OVERRIDES.items():
        if hasattr(theme, key):
            current = tuple(getattr(theme, key))
            # Theme colors can be RGBA: keep the original alpha
            setattr(theme, key, (*color, current[3]) if len(current) == 4 else color)


def _restore_theme(snapshot):
    """Write the saved palette back and delete the on-disk backup."""
    theme = _theme()
    for key, value in snapshot.items():
        try:
            setattr(theme, key, value)
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        os.remove(_backup_path())
    except OSError:
        pass


def _recover_from_backup():
    """If a backup file is left over (Blender closed / add-on reloaded while Post mode was on), restore it."""
    try:
        with open(_backup_path()) as f:
            snap = json.load(f)
    except (OSError, ValueError):
        return
    _restore_theme(snap)


def _recover_timer():
    if _saved is None:
        _recover_from_backup()
        _tag_redraw_all()
    return None


# ----------------------------------------------------------------------------
# Light drawing (per-type colors)
# ----------------------------------------------------------------------------
# Blender's theme has a single "Light" color, so lights are re-drawn on top in a
# viewport overlay with a color per light type.
_draw_handle = None


def _circle(cx, cy, r, steps=32):
    return [
        (cx + r * math.cos(2 * math.pi * i / steps), cy + r * math.sin(2 * math.pi * i / steps))
        for i in range(steps + 1)
    ]


def _draw_lights():
    saved = _saved
    if saved is None:
        return
    ctx = bpy.context
    area, region, rv3d, space = ctx.area, ctx.region, ctx.region_data, ctx.space_data
    if area is None or rv3d is None or area.as_pointer() != saved["area_ptr"]:
        return

    line_shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    fill_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    size = (region.width, region.height)
    gpu.state.blend_set('ALPHA')

    def strip(pts, color, width=2.0):
        if len(pts) < 2:
            return
        batch = batch_for_shader(line_shader, 'LINE_STRIP', {"pos": pts})
        line_shader.bind()
        line_shader.uniform_float("color", (*color, 1.0))
        line_shader.uniform_float("viewportSize", size)
        line_shader.uniform_float("lineWidth", width)
        batch.draw(line_shader)

    def dot(cx, cy, r, color):
        batch = batch_for_shader(fill_shader, 'TRI_FAN', {"pos": [(cx, cy)] + _circle(cx, cy, r, 16)})
        fill_shader.bind()
        fill_shader.uniform_float("color", (*color, 1.0))
        batch.draw(fill_shader)

    def project(p):
        return view3d_utils.location_3d_to_region_2d(region, rv3d, p)

    for obj in ctx.view_layer.objects:
        if obj.type != 'LIGHT':
            continue
        try:
            if not obj.visible_get(viewport=space):
                continue
        except (TypeError, RuntimeError):
            pass

        light = obj.data
        color = SELECT_COLOR if obj.select_get() else LIGHT_COLORS.get(light.type, (1.0, 1.0, 1.0))
        mw = obj.matrix_world
        center = project(mw.translation)
        if center is None:
            continue
        cx, cy = center

        # Screen-space direction of the light (-Z axis) for spot and sun
        direction = None
        if light.type in {'SPOT', 'SUN'}:
            tip = project(mw.translation + (mw.to_3x3() @ Vector((0.0, 0.0, -1.0))).normalized())
            if tip is not None:
                d = tip - center
                if d.length > 1e-4:
                    direction = d.normalized()

        if light.type == 'AREA':
            sx = light.size * 0.5
            sy = (light.size_y if light.shape in {'RECTANGLE', 'ELLIPSE'} else light.size) * 0.5
            if light.shape in {'DISK', 'ELLIPSE'}:
                local = [
                    Vector((sx * math.cos(a), sy * math.sin(a), 0.0))
                    for a in (2 * math.pi * i / 32 for i in range(33))
                ]
            else:
                local = [Vector(v) for v in ((-sx, -sy, 0), (sx, -sy, 0), (sx, sy, 0), (-sx, sy, 0), (-sx, -sy, 0))]
            pts = [project(mw @ v) for v in local]
            if all(p is not None for p in pts):
                strip([tuple(p) for p in pts], color)
            dot(cx, cy, 3.0, color)
        elif light.type == 'SUN':
            strip(_circle(cx, cy, 8), color)
            for i in range(8):
                a = 2 * math.pi * i / 8
                strip([(cx + 11 * math.cos(a), cy + 11 * math.sin(a)),
                       (cx + 17 * math.cos(a), cy + 17 * math.sin(a))], color)
            if direction:
                strip([(cx, cy), (cx + direction.x * 45, cy + direction.y * 45)], color)
        else:  # POINT, SPOT
            strip(_circle(cx, cy, 10), color)
            dot(cx, cy, 3.0, color)
            if light.type == 'SPOT' and direction:
                strip([(cx, cy), (cx + direction.x * 40, cy + direction.y * 40)], color)

    gpu.state.blend_set('NONE')


def _tag_redraw_all():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            area.tag_redraw()


def exit_post_mode():
    """Restore everything. Safe to call when Post mode is not active."""
    global _saved
    saved, _saved = _saved, None
    if not saved:
        return

    _restore_theme(saved["theme"])

    win, area = _find_area(saved["area_ptr"])
    space = _view3d_space(area) if area else None
    if space:
        # Leave local view first, then restore our saved settings on top
        if saved.get("local_view") and space.local_view is not None:
            try:
                with bpy.context.temp_override(
                    window=win, area=area, region=_window_region(area), space_data=space
                ):
                    bpy.ops.view3d.localview(frame_selected=False)
            except RuntimeError:
                pass

        r3d = space.region_3d
        persp, loc, rot, dist = saved["view"]
        r3d.view_location = loc
        r3d.view_rotation = rot
        r3d.view_distance = dist
        r3d.view_perspective = persp

        space.shading.type = saved["shading_type"]
        space.shading.wireframe_color_type = saved["wire_color"]

        for name, value in saved["types"].items():
            setattr(space, name, value)
        for name, value in saved["overlay"].items():
            setattr(space.overlay, name, value)

    _tag_redraw_all()


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------
class POSTMODE_OT_enter(bpy.types.Operator):
    """Turn this 3D viewport into a top-down wireframe view of cameras and lights"""
    bl_idname = "post_mode.enter"
    bl_label = "Enter Post mode"

    @classmethod
    def poll(cls, context):
        return (
            _saved is None
            and context.area is not None
            and context.area.type == 'VIEW_3D'
        )

    def execute(self, context):
        global _saved
        area = context.area
        space = area.spaces.active
        r3d = space.region_3d
        overlay = space.overlay
        collection = context.scene.post_mode_collection

        _recover_from_backup()  # clean up after an earlier session that never exited Post mode

        if collection and space.local_view is not None:
            self.report({'ERROR'}, "Exit Local View first (or clear the collection)")
            return {'CANCELLED'}

        saved = {
            "area_ptr": area.as_pointer(),
            "shading_type": space.shading.type,
            "wire_color": space.shading.wireframe_color_type,
            "view": (
                r3d.view_perspective,
                r3d.view_location.copy(),
                r3d.view_rotation.copy(),
                r3d.view_distance,
            ),
            "types": {},
            "overlay": {},
            "theme": _snapshot_theme(),
            "local_view": False,
        }
        _saved = saved

        try:
            self._enter(context, area, space, r3d, overlay, saved, collection)
        except Exception:
            # Roll back whatever was changed so the viewport isn't left half-modified
            exit_post_mode()
            raise
        return {'FINISHED'}

    def _enter(self, context, area, space, r3d, overlay, saved, collection):
        # Object-type visibility (per viewport). Default: keep only cameras + lights.
        # With a collection set, all types are enabled and Local View does the isolating.
        for name in dir(space):
            if name.startswith("show_object_viewport_"):
                saved["types"][name] = getattr(space, name)
                if collection:
                    setattr(space, name, True)
                else:
                    setattr(space, name, name[len("show_object_viewport_"):] in VISIBLE_TYPES)

        # Overlays: make sure extras (cameras/lights) are drawn with theme colors
        for name, value in (("show_extras", True), ("show_light_colors", False)):
            if hasattr(overlay, name):
                saved["overlay"][name] = getattr(overlay, name)
                setattr(overlay, name, value)

        # Wireframe shading, colored by theme (we override the theme colors below)
        space.shading.type = 'WIREFRAME'
        space.shading.wireframe_color_type = 'THEME'

        if collection:
            self._isolate(context, area, space, collection, saved)

        # Top orthographic view
        r3d.view_perspective = 'ORTHO'
        r3d.view_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))

        _apply_theme_overrides()

        # Frame the scene
        region = _window_region(area)
        if region:
            try:
                with context.temp_override(area=area, region=region, space_data=space):
                    bpy.ops.view3d.view_all(center=False)
            except RuntimeError:
                pass

        _tag_redraw_all()


    def _isolate(self, context, area, space, collection, saved):
        """Show only cameras, lights and the collection's objects in this viewport (Local View)."""
        view_layer = context.view_layer
        objects = view_layer.objects

        orig_selected = [o.name for o in objects if o.select_get()]
        orig_active = objects.active.name if objects.active else None

        targets = {o.name for o in objects if o.type in {'CAMERA', 'LIGHT'}}
        targets |= {o.name for o in collection.all_objects if o.name in objects}

        def set_select(obj, state):
            try:
                obj.select_set(state)
            except RuntimeError:
                pass

        for o in objects:
            set_select(o, False)
        for name in targets:
            set_select(objects[name], True)

        region = _window_region(area)
        try:
            with context.temp_override(window=context.window, area=area, region=region, space_data=space):
                bpy.ops.view3d.localview(frame_selected=False)
        except RuntimeError as e:
            self.report({'WARNING'}, f"Could not isolate collection: {e}")

        saved["local_view"] = space.local_view is not None

        # Put the user's original selection back (as far as Local View allows)
        for o in objects:
            set_select(o, False)
        for name in orig_selected:
            if name in objects:
                set_select(objects[name], True)
        if orig_active and orig_active in objects:
            objects.active = objects[orig_active]


class POSTMODE_OT_exit(bpy.types.Operator):
    """Restore the viewport and theme colors"""
    bl_idname = "post_mode.exit"
    bl_label = "Exit Post mode"

    @classmethod
    def poll(cls, context):
        return _saved is not None

    def execute(self, context):
        exit_post_mode()
        return {'FINISHED'}


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
class POSTMODE_PT_panel(bpy.types.Panel):
    bl_label = "Post Mode"
    bl_idname = "POSTMODE_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Post"

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.enabled = _saved is None
        col.prop(context.scene, "post_mode_collection", text="Keep Visible")
        if _saved is None:
            layout.operator("post_mode.enter", icon='LIGHT')
        else:
            layout.operator("post_mode.exit", icon='X')


# ----------------------------------------------------------------------------
# Registration
# ----------------------------------------------------------------------------
@persistent
def _on_load_pre(*_args):
    # Don't leave modified theme colors behind when another file is opened
    global _saved
    if _saved:
        _restore_theme(_saved["theme"])
        _saved = None


classes = (POSTMODE_OT_enter, POSTMODE_OT_exit, POSTMODE_PT_panel)


def register():
    global _draw_handle
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_lights, (), 'WINDOW', 'POST_PIXEL')
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.post_mode_collection = bpy.props.PointerProperty(
        name="Keep Visible",
        description="Objects in this collection (including child collections) stay visible in Post mode, "
                    "in addition to cameras and lights",
        type=bpy.types.Collection,
    )
    bpy.app.handlers.load_pre.append(_on_load_pre)
    # Undo any theme override left behind by a previous session / reload
    bpy.app.timers.register(_recover_timer, first_interval=0.5)


def unregister():
    global _draw_handle
    exit_post_mode()
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    if _on_load_pre in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_on_load_pre)
    del bpy.types.Scene.post_mode_collection
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

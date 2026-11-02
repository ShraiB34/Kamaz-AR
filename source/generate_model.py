#!/usr/bin/env python3
"""
Photorealistic glTF 2.0 (.glb) Generator v2.0
EMERCOM Mobile Communications Hub on KAMAZ-43118 6x6 Chassis (0828 MM 18)

Features:
- Real perspective-rectified photographic textures from the 13 walkaround drone photos
  (Albedo + Occlusion/Roughness/Metallic ORM + Tangent-Space Normal maps).
- High-poly smooth-shaded sculpted geometry:
  - Toroidal KAMA-1260 425/85 R21 off-road tires with 36 3D shoulder tread lugs & 10-bolt planetary hubs
  - Rounded-corner sculpted KAMAZ-43118 cab with flared wheel arches, recessed footsteps, 3D wipers,
    curved smoked sun visor, clearance lights, spotlights, tubular mirror assemblies, and dual-dome blue lightbar
  - Detailed 6x6 chassis with 3 differentials, leaf springs, tandem curved mudguards, brushed-aluminum fuel tank,
    cyclone air cleaner & snorkel, air receivers, toolboxes, and upright spare wheel carrier
  - Beveled KUNG shelter with 3D window frames, front cylindrical tent roll, recessed illuminated telecom room,
    recessed rear technical bay with 3D cable reels, perforated folding stairs, grated rear ramp & ladder
  - Roof complex with real ribbed walkway texture, tubular grab rails, beveled roof hatches, slatted SUPER SKYRIDER
    expedition rack, telescopic pneumatic mast + PTZ camera + coiled RF cable, and curved parabolic iNetVu VSAT dish.
"""

import io
import json
import math
import os
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extract_photo_textures import generate_pbr_dict, FONT_BOLD


# =============================================================================
# 1. MESH DATA & SMOOTH / BEVELED PROCEDURAL GEOMETRY HELPERS
# =============================================================================
class MeshData:
    def __init__(self):
        self.positions = []
        self.normals = []
        self.uvs = []
        self.indices = []

    def add_vertex(self, pos, norm, uv=(0.0, 0.0)):
        idx = len(self.positions)
        self.positions.append((float(pos[0]), float(pos[1]), float(pos[2])))
        n = np.array(norm, dtype=np.float64)
        nl = np.linalg.norm(n)
        if nl > 1e-9:
            n /= nl
        else:
            n = np.array([0.0, 1.0, 0.0])
        self.normals.append((float(n[0]), float(n[1]), float(n[2])))
        self.uvs.append((float(uv[0]), float(uv[1])))
        return idx

    def add_quad(self, p0, p1, p2, p3, uv0=(0, 1), uv1=(1, 1), uv2=(1, 0), uv3=(0, 0), normal=None):
        v1 = np.array(p1, dtype=np.float64) - np.array(p0, dtype=np.float64)
        v2 = np.array(p3, dtype=np.float64) - np.array(p0, dtype=np.float64)
        cross = np.cross(v1, v2)
        cl = np.linalg.norm(cross)
        if normal is None:
            normal = (cross / cl) if cl > 1e-9 else (0.0, 1.0, 0.0)
        i0 = self.add_vertex(p0, normal, uv0)
        i1 = self.add_vertex(p1, normal, uv1)
        i2 = self.add_vertex(p2, normal, uv2)
        i3 = self.add_vertex(p3, normal, uv3)
        if np.dot(cross, np.array(normal, dtype=np.float64)) < 0.0:
            self.indices.extend([i0, i2, i1, i0, i3, i2])
        else:
            self.indices.extend([i0, i1, i2, i0, i2, i3])

    def add_smooth_quad(self, p0, p1, p2, p3, n0, n1, n2, n3, uv0=(0, 1), uv1=(1, 1), uv2=(1, 0), uv3=(0, 0)):
        v1 = np.array(p1, dtype=np.float64) - np.array(p0, dtype=np.float64)
        v2 = np.array(p3, dtype=np.float64) - np.array(p0, dtype=np.float64)
        cross = np.cross(v1, v2)
        avg_n = np.array(n0, dtype=np.float64) + np.array(n1, dtype=np.float64) + np.array(n2, dtype=np.float64) + np.array(n3, dtype=np.float64)
        i0 = self.add_vertex(p0, n0, uv0)
        i1 = self.add_vertex(p1, n1, uv1)
        i2 = self.add_vertex(p2, n2, uv2)
        i3 = self.add_vertex(p3, n3, uv3)
        if np.dot(cross, avg_n) < 0.0:
            self.indices.extend([i0, i2, i1, i0, i3, i2])
        else:
            self.indices.extend([i0, i1, i2, i0, i2, i3])

    def add_tri(self, p0, p1, p2, uv0=(0, 0), uv1=(1, 0), uv2=(0.5, 1), normal=None):
        if normal is None:
            v1 = np.array(p1, dtype=np.float64) - np.array(p0, dtype=np.float64)
            v2 = np.array(p2, dtype=np.float64) - np.array(p0, dtype=np.float64)
            n = np.cross(v1, v2)
            nl = np.linalg.norm(n)
            normal = (n / nl) if nl > 1e-9 else (0.0, 1.0, 0.0)
        i0 = self.add_vertex(p0, normal, uv0)
        i1 = self.add_vertex(p1, normal, uv1)
        i2 = self.add_vertex(p2, normal, uv2)
        self.indices.extend([i0, i1, i2])

    def merge(self, other, offset=(0.0, 0.0, 0.0), rot_x_deg=0.0, rot_y_deg=0.0, rot_z_deg=0.0):
        rx, ry, rz = math.radians(rot_x_deg), math.radians(rot_y_deg), math.radians(rot_z_deg)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        off = np.array(offset, dtype=np.float64)
        base = len(self.positions)
        for p, n, uv in zip(other.positions, other.normals, other.uvs):
            pr = R @ np.array(p, dtype=np.float64) + off
            nr = R @ np.array(n, dtype=np.float64)
            self.positions.append((float(pr[0]), float(pr[1]), float(pr[2])))
            self.normals.append((float(nr[0]), float(nr[1]), float(nr[2])))
            self.uvs.append(uv)
        for idx in other.indices:
            self.indices.append(base + idx)


def make_box(sx, sy, sz, center=(0, 0, 0)):
    m = MeshData()
    cx, cy, cz = center
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    m.add_quad((cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz), (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz), normal=(0, 0, 1))
    m.add_quad((cx + hx, cy - hy, cz - hz), (cx - hx, cy - hy, cz - hz), (cx - hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz - hz), normal=(0, 0, -1))
    m.add_quad((cx + hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy + hy, cz - hz), (cx + hx, cy + hy, cz + hz), normal=(1, 0, 0))
    m.add_quad((cx - hx, cy - hy, cz - hz), (cx - hx, cy - hy, cz + hz), (cx - hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz - hz), normal=(-1, 0, 0))
    m.add_quad((cx - hx, cy + hy, cz + hz), (cx + hx, cy + hy, cz + hz), (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz), normal=(0, 1, 0))
    m.add_quad((cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz + hz), (cx - hx, cy - hy, cz + hz), normal=(0, -1, 0))
    return m


def make_beveled_box(sx, sy, sz, bevel=0.025, center=(0, 0, 0)):
    """Smooth-edged beveled rectangular block for realistic metallic specular highlights on corners."""
    m = MeshData()
    cx, cy, cz = center
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    b = min(bevel, hx * 0.35, hy * 0.35, hz * 0.35)
    # Build 4-slice rounded profile along Y with 8-gon rounded rectangle in XZ!
    # Corner offsets in XZ:
    corners = []
    for sx_s, sz_s, a0 in [(+1, +1, 0.0), (-1, +1, math.pi*0.5), (-1, -1, math.pi), (+1, -1, math.pi*1.5)]:
        for sub in [0.15, 0.85]:
            ang = a0 + sub * (math.pi * 0.5)
            px = sx_s * (hx - b) + b * math.cos(ang)
            pz = sz_s * (hz - b) + b * math.sin(ang)
            nx = math.cos(ang)
            nz = math.sin(ang)
            corners.append((px, pz, nx, nz))
    y_rings = [
        (-hy,       0.15, -0.98),
        (-hy + b,   1.00, -0.15),
        (+hy - b,   1.00, +0.15),
        (+hy,       0.15, +0.98),
    ]
    N = len(corners)
    for r_i in range(len(y_rings) - 1):
        y0, sc0, ny0 = y_rings[r_i]
        y1, sc1, ny1 = y_rings[r_i + 1]
        for k in range(N):
            k2 = (k + 1) % N
            x0a = cx + corners[k][0] * (1.0 if sc0 > 0.5 else (hx - b) / max(hx, 1e-5))
            z0a = cz + corners[k][1] * (1.0 if sc0 > 0.5 else (hz - b) / max(hz, 1e-5))
            x0b = cx + corners[k2][0] * (1.0 if sc0 > 0.5 else (hx - b) / max(hx, 1e-5))
            z0b = cz + corners[k2][1] * (1.0 if sc0 > 0.5 else (hz - b) / max(hz, 1e-5))
            x1a = cx + corners[k][0] * (1.0 if sc1 > 0.5 else (hx - b) / max(hx, 1e-5))
            z1a = cz + corners[k][1] * (1.0 if sc1 > 0.5 else (hz - b) / max(hz, 1e-5))
            x1b = cx + corners[k2][0] * (1.0 if sc1 > 0.5 else (hx - b) / max(hx, 1e-5))
            z1b = cz + corners[k2][1] * (1.0 if sc1 > 0.5 else (hz - b) / max(hz, 1e-5))
            n0a = (corners[k][2] * sc0, ny0, corners[k][3] * sc0)
            n0b = (corners[k2][2] * sc0, ny0, corners[k2][3] * sc0)
            n1a = (corners[k][2] * sc1, ny1, corners[k][3] * sc1)
            n1b = (corners[k2][2] * sc1, ny1, corners[k2][3] * sc1)
            m.add_smooth_quad((x0a, cy + y0, z0a), (x0b, cy + y0, z0b), (x1b, cy + y1, z1b), (x1a, cy + y1, z1a), n0a, n0b, n1b, n1a)
    # Top & bottom caps
    top_c = (cx, cy + hy, cz)
    bot_c = (cx, cy - hy, cz)
    for k in range(N):
        k2 = (k + 1) % N
        r_scx = (hx - b) / max(hx, 1e-5)
        r_scz = (hz - b) / max(hz, 1e-5)
        p_t0 = (cx + corners[k][0] * r_scx, cy + hy, cz + corners[k][1] * r_scz)
        p_t1 = (cx + corners[k2][0] * r_scx, cy + hy, cz + corners[k2][1] * r_scz)
        m.add_tri(top_c, p_t0, p_t1, normal=(0, 1, 0))
        p_b0 = (cx + corners[k][0] * r_scx, cy - hy, cz + corners[k][1] * r_scz)
        p_b1 = (cx + corners[k2][0] * r_scx, cy - hy, cz + corners[k2][1] * r_scz)
        m.add_tri(bot_c, p_b1, p_b0, normal=(0, -1, 0))
    return m


def make_cylinder(r_bot, r_top, length, segs=24, axis="y", center=(0, 0, 0), caps=True):
    m = MeshData()
    cx, cy, cz = center
    hl = length * 0.5
    angles = [2.0 * math.pi * i / segs for i in range(segs + 1)]

    def to_axis(u, v, w):
        if axis == "y":
            return (cx + u, cy + w, cz + v)
        elif axis == "x":
            return (cx + w, cy + u, cz + v)
        else:
            return (cx + u, cy + v, cz + w)

    def n_axis(nu, nv, nw):
        if axis == "y":
            return (nu, nw, nv)
        elif axis == "x":
            return (nw, nu, nv)
        else:
            return (nu, nv, nw)

    slope = (r_bot - r_top) / max(length, 1e-6)
    for i in range(segs):
        a0, a1 = angles[i], angles[i + 1]
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        p00 = to_axis(r_bot * c0, r_bot * s0, -hl)
        p10 = to_axis(r_bot * c1, r_bot * s1, -hl)
        p11 = to_axis(r_top * c1, r_top * s1, +hl)
        p01 = to_axis(r_top * c0, r_top * s0, +hl)
        n0 = n_axis(c0, s0, slope)
        n1 = n_axis(c1, s1, slope)
        u0, u1 = i / segs, (i + 1) / segs
        m.add_smooth_quad(p00, p01, p11, p10, n0, n0, n1, n1, uv0=(u0, 1), uv1=(u0, 0), uv2=(u1, 0), uv3=(u1, 1))

    if caps:
        if r_top > 1e-5:
            ct = to_axis(0, 0, +hl)
            nt = n_axis(0, 0, 1)
            for i in range(segs):
                a0, a1 = angles[i], angles[i + 1]
                p0 = to_axis(r_top * math.cos(a0), r_top * math.sin(a0), +hl)
                p1 = to_axis(r_top * math.cos(a1), r_top * math.sin(a1), +hl)
                m.add_tri(ct, p1, p0, uv0=(0.5, 0.5), uv1=(0.5 + 0.5*math.cos(a1), 0.5 + 0.5*math.sin(a1)), uv2=(0.5 + 0.5*math.cos(a0), 0.5 + 0.5*math.sin(a0)), normal=nt)
        if r_bot > 1e-5:
            cb = to_axis(0, 0, -hl)
            nb = n_axis(0, 0, -1)
            for i in range(segs):
                a0, a1 = angles[i], angles[i + 1]
                p0 = to_axis(r_bot * math.cos(a0), r_bot * math.sin(a0), -hl)
                p1 = to_axis(r_bot * math.cos(a1), r_bot * math.sin(a1), -hl)
                m.add_tri(cb, p0, p1, uv0=(0.5, 0.5), uv1=(0.5 + 0.5*math.cos(a0), 0.5 + 0.5*math.sin(a0)), uv2=(0.5 + 0.5*math.cos(a1), 0.5 + 0.5*math.sin(a1)), normal=nb)
    return m


# =============================================================================
# 2. HIGH-POLY TOROIDAL OFF-ROAD WHEEL (KAMA-1260 425/85 R21)
# =============================================================================
def build_photoreal_wheel(outer_side="+x"):
    """
    Returns (md_face, md_tread, md_bolts) centered at (0,0,0) with axle along X.
    outer_side='+x' means +X faces outward (left wheels); '-x' means -X faces outward (right wheels).
    """
    sign = 1.0 if outer_side == "+x" else -1.0
    md_face = MeshData()
    md_tread = MeshData()
    md_bolts = MeshData()

    segs = 40
    angles = [2.0 * math.pi * i / segs for i in range(segs + 1)]

    # Cross-section along X (from outer hub center out to tread, then across tread to inner sidewall):
    # Profile of OUTER face (mapped with photographic wheel_face!):
    # (x_local * sign, radius, nx_local * sign, nr)
    outer_prof = [
        (0.165, 0.000, 1.0,  0.00),  # Hub cap center
        (0.165, 0.095, 0.95, 0.30),  # Planetary hub face
        (0.132, 0.112, 0.45, 0.89),  # Hub step down
        (0.125, 0.215, 0.96, 0.28),  # Bolt circle flange
        (0.088, 0.248, 0.55, 0.83),  # Deep concave wheel dish
        (0.155, 0.285, 0.85, -0.52), # Rim bead lip
        (0.198, 0.385, 0.92, 0.38),  # Lower sidewall bulge
        (0.206, 0.475, 0.95, 0.31),  # Max tire sidewall bulge!
        (0.172, 0.532, 0.62, 0.78),  # Rounded shoulder into tread
    ]

    for j in range(len(outer_prof) - 1):
        x0, r0, nx0, nr0 = outer_prof[j]
        x1, r1, nx1, nr1 = outer_prof[j + 1]
        for i in range(segs):
            a0, a1 = angles[i], angles[i + 1]
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)

            p00 = (sign * x0, r0 * s0, r0 * c0)
            p01 = (sign * x1, r1 * s0, r1 * c0)
            p11 = (sign * x1, r1 * s1, r1 * c1)
            p10 = (sign * x0, r0 * s1, r0 * c1)

            n00 = (sign * nx0, nr0 * s0, nr0 * c0)
            n01 = (sign * nx1, nr1 * s0, nr1 * c0)
            n11 = (sign * nx1, nr1 * s1, nr1 * c1)
            n10 = (sign * nx0, nr0 * s1, nr0 * c1)

            # Planar polar UV mapping onto wheel_face (radius 0..0.542 -> UV radius 0..0.48)
            def w_uv(r, c, s):
                u = 0.5 + sign * (r / 0.545) * 0.475 * c
                v = 0.5 - (r / 0.545) * 0.475 * s
                return (u, v)

            uv00, uv01 = w_uv(r0, c0, s0), w_uv(r1, c0, s0)
            uv11, uv10 = w_uv(r1, c1, s1), w_uv(r0, c1, s1)
            if sign > 0:
                md_face.add_smooth_quad(p00, p10, p11, p01, n00, n10, n11, n01, uv00, uv10, uv11, uv01)
            else:
                md_face.add_smooth_quad(p00, p01, p11, p10, n00, n01, n11, n10, uv00, uv01, uv11, uv10)

    # Crowned Tread Band (from outer shoulder +0.172 across crown 0.545 to inner shoulder -0.172)
    tread_prof = [
        (+0.172, 0.532, +0.48, 0.88, 0.00),
        (+0.095, 0.544, +0.16, 0.99, 0.25),
        ( 0.000, 0.547,  0.00, 1.00, 0.50),
        (-0.095, 0.544, -0.16, 0.99, 0.75),
        (-0.172, 0.532, -0.48, 0.88, 1.00),
    ]
    for j in range(len(tread_prof) - 1):
        x0, r0, nx0, nr0, u_t0 = tread_prof[j]
        x1, r1, nx1, nr1, u_t1 = tread_prof[j + 1]
        for i in range(segs):
            a0, a1 = angles[i], angles[i + 1]
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            p00 = (sign * x0, r0 * s0, r0 * c0)
            p01 = (sign * x1, r1 * s0, r1 * c0)
            p11 = (sign * x1, r1 * s1, r1 * c1)
            p10 = (sign * x0, r0 * s1, r0 * c1)
            n00 = (sign * nx0, nr0 * s0, nr0 * c0)
            n01 = (sign * nx1, nr1 * s0, nr1 * c0)
            n11 = (sign * nx1, nr1 * s1, nr1 * c1)
            n10 = (sign * nx0, nr0 * s1, nr0 * c1)
            v0, v1 = (i / segs) * 4.0, ((i + 1) / segs) * 4.0  # Repeat tread 4x around circumference!
            if sign > 0:
                md_tread.add_smooth_quad(p00, p01, p11, p10, n00, n01, n11, n10, (u_t0, v0), (u_t1, v0), (u_t1, v1), (u_t0, v1))
            else:
                md_tread.add_smooth_quad(p00, p10, p11, p01, n00, n10, n11, n01, (u_t0, v0), (u_t0, v1), (u_t1, v1), (u_t1, v0))

    # Inner sidewall & inner steel rim closure
    inner_prof = [
        (-0.172, 0.532, -0.62, 0.78),
        (-0.202, 0.460, -0.95, 0.30),
        (-0.165, 0.285, -0.92, -0.35),
        (-0.115, 0.140, -1.00, 0.00),
        (-0.115, 0.000, -1.00, 0.00),
    ]
    for j in range(len(inner_prof) - 1):
        x0, r0, nx0, nr0 = inner_prof[j]
        x1, r1, nx1, nr1 = inner_prof[j + 1]
        for i in range(segs):
            a0, a1 = angles[i], angles[i + 1]
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            p00 = (sign * x0, r0 * s0, r0 * c0)
            p01 = (sign * x1, r1 * s0, r1 * c0)
            p11 = (sign * x1, r1 * s1, r1 * c1)
            p10 = (sign * x0, r0 * s1, r0 * c1)
            n00 = (sign * nx0, nr0 * s0, nr0 * c0)
            n01 = (sign * nx1, nr1 * s0, nr1 * c0)
            n11 = (sign * nx1, nr1 * s1, nr1 * c1)
            n10 = (sign * nx0, nr0 * s1, nr0 * c1)
            if sign > 0:
                md_tread.add_smooth_quad(p00, p01, p11, p10, n00, n01, n11, n10, (0.1, 0.1), (0.2, 0.1), (0.2, 0.2), (0.1, 0.2))
            else:
                md_tread.add_smooth_quad(p00, p10, p11, p01, n00, n10, n11, n01, (0.1, 0.1), (0.1, 0.2), (0.2, 0.2), (0.2, 0.1))

    # 36 Real 3D Sculpted Off-Road Shoulder Tread Lugs around the circumference!
    lug_box = make_box(0.115, 0.024, 0.046)
    for k in range(36):
        ang_deg = k * 10.0
        ang = math.radians(ang_deg)
        side_alt = 1.0 if (k % 2 == 0) else -1.0
        lx = sign * (0.125 * side_alt)
        ly = 0.542 * math.sin(ang)
        lz = 0.542 * math.cos(ang)
        md_tread.merge(lug_box, offset=(lx, ly, lz), rot_x_deg=-ang_deg)

    # 10 Metallic Lug Nuts + Hub Cap Bolts on outer planetary wheel hub!
    nut = make_cylinder(0.013, 0.011, 0.022, segs=6, axis="x")
    for k in range(10):
        ang = 2.0 * math.pi * k / 10.0
        md_bolts.merge(nut, offset=(sign * 0.134, 0.172 * math.sin(ang), 0.172 * math.cos(ang)))
    for k in range(6):
        ang = 2.0 * math.pi * k / 6.0
        md_bolts.merge(nut, offset=(sign * 0.170, 0.068 * math.sin(ang), 0.068 * math.cos(ang)))

    return md_face, md_tread, md_bolts


# =============================================================================
# 3. GLB BINARY BUILDER WITH FULL glTF 2.0 PBR SUPPORT (Albedo + ORM + Normal)
# =============================================================================
class GLBBuilder:
    def __init__(self):
        self.bin_data = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.images = []
        self.textures = []
        self.materials = []
        self.meshes = []
        self.nodes = []

    def _append_bytes(self, raw_bytes, target=None):
        while len(self.bin_data) % 4 != 0:
            self.bin_data.append(0)
        offset = len(self.bin_data)
        self.bin_data.extend(raw_bytes)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw_bytes)}
        if target is not None:
            bv["target"] = target
        idx = len(self.buffer_views)
        self.buffer_views.append(bv)
        return idx

    def add_rgb_texture(self, arr_uint8, name="tex", fmt="JPEG", quality=92):
        buf = io.BytesIO()
        pil = Image.fromarray(arr_uint8)
        if fmt == "JPEG":
            pil.save(buf, format="JPEG", quality=quality, subsampling=0)
            mime = "image/jpeg"
        else:
            pil.save(buf, format="PNG", optimize=True)
            mime = "image/png"
        bv_idx = self._append_bytes(buf.getvalue())
        img_idx = len(self.images)
        self.images.append({"bufferView": bv_idx, "mimeType": mime, "name": name})
        tex_idx = len(self.textures)
        self.textures.append({"source": img_idx, "sampler": 0})
        return tex_idx

    def add_pbr_tex_set(self, name, tex_entry, normal_scale=0.85, occ_strength=0.88, double_sided=False):
        t_alb = self.add_rgb_texture(tex_entry["albedo"], f"{name}_albedo", fmt="JPEG", quality=93)
        t_orm = self.add_rgb_texture(tex_entry["orm"],    f"{name}_orm",    fmt="JPEG", quality=90)
        t_nrm = self.add_rgb_texture(tex_entry["normal"], f"{name}_normal", fmt="JPEG", quality=90)
        mat = {
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": t_alb},
                "metallicRoughnessTexture": {"index": t_orm},
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
            },
            "occlusionTexture": {"index": t_orm, "strength": float(occ_strength)},
            "normalTexture": {"index": t_nrm, "scale": float(normal_scale)},
            "doubleSided": bool(double_sided),
        }
        idx = len(self.materials)
        self.materials.append(mat)
        return idx

    def add_solid_material(self, name, base_color=(0.85, 0.87, 0.90, 1.0), metallic=0.1, roughness=0.35,
                           emissive=None, alpha_blend=False, double_sided=False):
        pbr = {
            "baseColorFactor": [float(c) for c in base_color],
            "metallicFactor": float(metallic),
            "roughnessFactor": float(roughness),
        }
        mat = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": bool(double_sided)}
        if emissive is not None:
            mat["emissiveFactor"] = [float(c) for c in emissive]
        if alpha_blend:
            mat["alphaMode"] = "BLEND"
        idx = len(self.materials)
        self.materials.append(mat)
        return idx

    def add_mesh_primitives(self, name, prim_list):
        primitives = []
        for md, mat_idx in prim_list:
            if not md.positions or not md.indices:
                continue
            pos_arr = np.array(md.positions, dtype=np.float32)
            nrm_arr = np.array(md.normals, dtype=np.float32)
            uv_arr  = np.array(md.uvs, dtype=np.float32)
            idx_arr = np.array(md.indices, dtype=np.uint32)

            bv_pos = self._append_bytes(pos_arr.tobytes(), target=34962)
            bv_nrm = self._append_bytes(nrm_arr.tobytes(), target=34962)
            bv_uv  = self._append_bytes(uv_arr.tobytes(),  target=34962)
            bv_idx = self._append_bytes(idx_arr.tobytes(), target=34963)

            acc_pos = len(self.accessors)
            self.accessors.append({
                "bufferView": bv_pos, "componentType": 5126, "count": len(pos_arr), "type": "VEC3",
                "min": pos_arr.min(axis=0).tolist(), "max": pos_arr.max(axis=0).tolist(),
            })
            acc_nrm = len(self.accessors)
            self.accessors.append({"bufferView": bv_nrm, "componentType": 5126, "count": len(nrm_arr), "type": "VEC3"})
            acc_uv = len(self.accessors)
            self.accessors.append({"bufferView": bv_uv, "componentType": 5126, "count": len(uv_arr), "type": "VEC2"})
            acc_idx = len(self.accessors)
            self.accessors.append({"bufferView": bv_idx, "componentType": 5125, "count": len(idx_arr), "type": "SCALAR"})

            primitives.append({
                "attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm, "TEXCOORD_0": acc_uv},
                "indices": acc_idx,
                "material": mat_idx,
            })
        mesh_idx = len(self.meshes)
        self.meshes.append({"name": name, "primitives": primitives})
        return mesh_idx

    def add_node(self, name, mesh_idx=None, translation=None, rotation_quat=None, scale=None, children=None):
        node = {"name": name}
        if mesh_idx is not None:
            node["mesh"] = mesh_idx
        if translation is not None:
            node["translation"] = [float(v) for v in translation]
        if rotation_quat is not None:
            node["rotation"] = [float(v) for v in rotation_quat]
        if scale is not None:
            node["scale"] = [float(v) for v in scale]
        if children:
            node["children"] = list(children)
        idx = len(self.nodes)
        self.nodes.append(node)
        return idx

    def export_glb(self, out_path, root_nodes):
        while len(self.bin_data) % 4 != 0:
            self.bin_data.append(0)
        gltf = {
            "asset": {"version": "2.0", "generator": "Arena Photorealistic KAMAZ-43118 Builder v2.0"},
            "scene": 0,
            "scenes": [{"name": "KamazCommsHubScene", "nodes": root_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "textures": self.textures,
            "images": self.images,
            "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}],
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.bin_data)}],
        }
        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4 != 0:
            json_bytes += b" "
        total_len = 12 + 8 + len(json_bytes) + 8 + len(self.bin_data)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total_len))
            f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            f.write(json_bytes)
            f.write(struct.pack("<II", len(self.bin_data), 0x004E4942))
            f.write(self.bin_data)
        return total_len


# =============================================================================
# 4. ASSEMBLE PHOTOREALISTIC KAMAZ-43118 MOBILE COMMUNICATIONS HUB
# =============================================================================
def build_kamaz_comms_hub_glb(out_path):
    b = GLBBuilder()
    pbr = generate_pbr_dict()

    # Photo-projected PBR materials
    m_cab_front  = b.add_pbr_tex_set("CabFrontPBR",  pbr["cab_front"],  normal_scale=0.75, double_sided=True)
    m_cab_bumper = b.add_pbr_tex_set("CabBumperPBR", pbr["cab_bumper"], normal_scale=0.85, double_sided=True)
    m_cab_side_l = b.add_pbr_tex_set("CabSideLPBR",  pbr["cab_side_l"], normal_scale=0.75, double_sided=True)
    m_cab_side_r = b.add_pbr_tex_set("CabSideRPBR",  pbr["cab_side_r"], normal_scale=0.75, double_sided=True)
    m_wheel_face = b.add_pbr_tex_set("WheelFacePBR", pbr["wheel_face"], normal_scale=1.10)
    m_tire_tread = b.add_pbr_tex_set("TireTreadPBR", pbr["tire_tread"], normal_scale=1.35)
    m_kung_l     = b.add_pbr_tex_set("KungLeftPBR",  pbr["kung_left"],  normal_scale=0.70, double_sided=True)
    m_kung_r     = b.add_pbr_tex_set("KungRightPBR", pbr["kung_right"], normal_scale=0.70, double_sided=True)
    m_kung_rear  = b.add_pbr_tex_set("KungRearPBR",  pbr["kung_rear"],  normal_scale=0.75, double_sided=True)
    m_kung_roof  = b.add_pbr_tex_set("KungRoofPBR",  pbr["kung_roof"],  normal_scale=0.95, double_sided=True)
    m_vsat_dish  = b.add_pbr_tex_set("VsatDishPBR",  pbr["vsat_dish"],  normal_scale=0.75, double_sided=True)

    # Interior Server Rack & Rear Cable Bay texture
    rack_img = Image.new("RGB", (512, 512), (18, 22, 28))
    dr = ImageDraw.Draw(rack_img)
    for rx in [24, 185, 346]:
        dr.rectangle([rx, 24, rx + 142, 488], fill=(28, 34, 44), outline=(62, 72, 88), width=3)
        for ry in range(38, 470, 36):
            dr.rectangle([rx + 8, ry, rx + 134, ry + 26], fill=(14, 18, 24), outline=(44, 52, 64), width=1)
            for lx in range(rx + 16, rx + 82, 14):
                col = (34, 197, 94) if (lx + ry) % 3 else (56, 189, 248)
                dr.ellipse([lx, ry + 9, lx + 5, ry + 14], fill=col)
    dr.rectangle([188, 92, 322, 182], fill=(14, 58, 96), outline=(56, 189, 248), width=2)
    t_rack = b.add_rgb_texture(np.array(rack_img), "interior_racks", fmt="JPEG", quality=92)
    m_interior = len(b.materials)
    b.materials.append({
        "name": "InteriorRacks",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": t_rack}, "metallicFactor": 0.25, "roughnessFactor": 0.38},
        "emissiveTexture": {"index": t_rack},
        "emissiveFactor": [0.35, 0.35, 0.35],
    })

    # SUPER SKYRIDER fairing plate
    sr_img = Image.new("RGB", (512, 96), (20, 22, 26))
    dsr = ImageDraw.Draw(sr_img)
    dsr.rectangle([4, 4, 507, 91], outline=(55, 60, 70), width=3)
    dsr.text((88, 24), "SUPER SKYRIDER", fill=(215, 220, 228), font=ImageFont.truetype(FONT_BOLD, 36))
    t_sr = b.add_rgb_texture(np.array(sr_img), "skyrider", fmt="JPEG", quality=92)
    m_skyrider = len(b.materials)
    b.materials.append({
        "name": "SkyriderPlate",
        "pbrMetallicRoughness": {"baseColorTexture": {"index": t_sr}, "metallicFactor": 0.35, "roughnessFactor": 0.42},
        "doubleSided": True,
    })

    # Physically calibrated automotive & industrial materials (NOT chalky 0.98 white!)
    m_white_paint = b.add_solid_material("WhiteEnamel",    base_color=(0.835, 0.850, 0.872, 1.0), metallic=0.06, roughness=0.23, double_sided=True)
    m_chassis_blk = b.add_solid_material("ChassisSteel",   base_color=(0.065, 0.072, 0.082, 1.0), metallic=0.42, roughness=0.52)
    m_matte_plast = b.add_solid_material("MatteThermopl",  base_color=(0.055, 0.060, 0.068, 1.0), metallic=0.12, roughness=0.68, double_sided=True)
    m_brushed_alu = b.add_solid_material("BrushedAlum",    base_color=(0.74,  0.76,  0.79,  1.0), metallic=0.86, roughness=0.28)
    m_chrome      = b.add_solid_material("ChromeMirror",   base_color=(0.88,  0.90,  0.93,  1.0), metallic=0.96, roughness=0.08)
    m_dark_glass  = b.add_solid_material("SmokedGlass",    base_color=(0.04,  0.06,  0.09,  1.0), metallic=0.30, roughness=0.04)
    m_visor_smoke = b.add_solid_material("SunVisorAcrylic",base_color=(0.06,  0.04,  0.05,  0.90),metallic=0.20, roughness=0.10, alpha_blend=True, double_sided=True)
    m_beacon_blue = b.add_solid_material("BeaconBlue",     base_color=(0.05,  0.34,  0.88,  0.88),metallic=0.15, roughness=0.08, emissive=(0.04, 0.22, 0.65), alpha_blend=True)
    m_amber_lamp  = b.add_solid_material("AmberMarker",    base_color=(0.96,  0.52,  0.04,  1.0), metallic=0.10, roughness=0.16, emissive=(0.45, 0.18, 0.0))
    m_tail_red    = b.add_solid_material("TaillightRed",   base_color=(0.76,  0.06,  0.06,  1.0), metallic=0.15, roughness=0.14, emissive=(0.25, 0.02, 0.02))
    m_orange_acc  = b.add_solid_material("EmercomOrange",  base_color=(0.90,  0.38,  0.08,  1.0), metallic=0.08, roughness=0.28)
    m_blue_acc    = b.add_solid_material("EmercomBlue",    base_color=(0.09,  0.38,  0.70,  1.0), metallic=0.08, roughness=0.28)
    m_inner_door  = b.add_solid_material("DoorInsulation", base_color=(0.62,  0.64,  0.65,  1.0), metallic=0.08, roughness=0.55)
    m_safety_yel  = b.add_solid_material("SafetyYellow",   base_color=(0.92,  0.72,  0.08,  1.0), metallic=0.10, roughness=0.35)

    child_nodes = []

    # -------------------------------------------------------------------------
    # A. 6x6 CHASSIS, AXLES, MUDGUARDS & BEHIND-CAB EQUIPMENT
    # -------------------------------------------------------------------------
    md_ch_blk = MeshData()
    md_ch_alu = MeshData()
    md_ch_wht = MeshData()
    md_ch_amb = MeshData()

    # Twin longitudinal C-channel frame rails + crossmembers + dark underbody AO shield
    for sx in [-0.43, +0.43]:
        md_ch_blk.merge(make_box(0.09, 0.25, 8.38, center=(sx, 0.90, -0.22)))
    for cz in [-4.25, -2.90, -1.55, -0.20, 1.35, 2.85, 3.95]:
        md_ch_blk.merge(make_box(0.96, 0.17, 0.11, center=(0.0, 0.90, cz)))
    # Dark underbody floor pan under KUNG so wheel wells have realistic deep shadow (no sky bleed underneath!)
    md_ch_blk.merge(make_box(2.26, 0.08, 5.72, center=(0.0, 1.13, -1.56)))

    # 3 Heavy-duty 6x6 drive axles + differential pumpkins + leaf springs + driveshafts!
    for idx_ax, az in enumerate([+3.02, -1.68, -3.02]):
        md_ch_blk.merge(make_cylinder(0.072, 0.072, 2.04, segs=18, axis="x", center=(0.0, 0.545, az)))
        # Differential pumpkin housing
        diff_x = 0.12 if idx_ax == 0 else 0.0
        md_ch_blk.merge(make_cylinder(0.185, 0.175, 0.28, segs=20, axis="z", center=(diff_x, 0.555, az)))
        # Leaf spring packs left & right
        for sx in [-0.54, +0.54]:
            md_ch_blk.merge(make_box(0.09, 0.08, 1.18, center=(sx, 0.72, az)))
    # Driveshafts connecting transfer case (z=0.85) to front & rear axles
    md_ch_blk.merge(make_box(0.38, 0.34, 0.44, center=(0.0, 0.76, 0.85)))  # Transfer case (раздатка)
    md_ch_blk.merge(make_cylinder(0.042, 0.042, 2.15, segs=12, axis="z", center=(0.06, 0.65, 1.94)))
    md_ch_blk.merge(make_cylinder(0.042, 0.042, 2.50, segs=12, axis="z", center=(0.0, 0.65, -0.42)))
    md_ch_blk.merge(make_cylinder(0.042, 0.042, 1.34, segs=12, axis="z", center=(0.0, 0.56, -2.35)))

    # Curved Tandem Thermoplastic Mudguards (подкрылки over z=-1.68 & z=-3.02) + Rubber Mudflaps!
    for sx_s in [-1.0, +1.0]:
        xc = sx_s * 0.99
        md_ch_blk.merge(make_box(0.48, 0.035, 1.72), offset=(xc, 1.14, -2.35))
        md_ch_blk.merge(make_box(0.48, 0.52, 0.035), offset=(xc, 0.92, -1.36), rot_x_deg=-28)
        md_ch_blk.merge(make_box(0.48, 0.56, 0.035), offset=(xc, 0.90, -3.34), rot_x_deg=+28)
        md_ch_blk.merge(make_box(0.46, 0.42, 0.018), offset=(xc, 0.48, -3.52))

    # Behind-Cab Left Equipment (+X, photo_6):
    # 1) Brushed aluminum rectangular fuel tank with rounded corners & black steel straps (z=1.78)
    md_ch_alu.merge(make_beveled_box(0.58, 0.54, 0.78, bevel=0.055, center=(+0.84, 0.80, 1.84)))
    for sz_st in [1.62, 2.06]:
        md_ch_blk.merge(make_box(0.60, 0.56, 0.032, center=(+0.84, 0.80, sz_st)))
    md_ch_alu.merge(make_cylinder(0.045, 0.045, 0.08, segs=14, axis="y", center=(+0.98, 1.09, 2.08)))
    # 2) Black battery locker & ribbed cylindrical air receiver (photo_6: z=0.68 & z=-0.05)
    md_ch_blk.merge(make_beveled_box(0.54, 0.46, 0.82, bevel=0.03, center=(+0.84, 0.84, 0.72)))
    md_ch_alu.merge(make_box(0.56, 0.47, 0.025, center=(+0.84, 0.84, 0.52)))
    md_ch_alu.merge(make_box(0.56, 0.47, 0.025, center=(+0.84, 0.84, 0.92)))
    md_ch_blk.merge(make_cylinder(0.155, 0.155, 0.52, segs=18, axis="y", center=(+0.86, 0.84, -0.02)))
    # 3) Cyclone air filter drum + rectangular upper snorkel behind cab left (z=2.06)
    md_ch_blk.merge(make_cylinder(0.20, 0.20, 0.48, segs=20, axis="x", center=(+0.86, 1.52, 2.06)))
    md_ch_blk.merge(make_cylinder(0.085, 0.085, 0.95, segs=16, axis="y", center=(+0.88, 2.15, 2.12)))
    md_ch_blk.merge(make_beveled_box(0.24, 0.54, 0.30, bevel=0.03, center=(+0.88, 2.66, 2.12)))

    # Behind-Cab Right Equipment (-X, photo_1):
    # Spare tire carrier frame + davit arm + right chassis equipment lockers
    md_ch_blk.merge(make_box(0.32, 0.95, 0.14, center=(-0.64, 1.48, 1.92)))
    md_ch_blk.merge(make_beveled_box(0.54, 0.48, 0.95, bevel=0.03, center=(-0.84, 0.82, 0.85)))
    md_ch_blk.merge(make_beveled_box(0.52, 0.46, 0.72, bevel=0.03, center=(-0.84, 0.82, -0.08)))
    # Side amber chassis marker LEDs
    for sx_m in [-1.14, +1.14]:
        md_ch_amb.merge(make_box(0.02, 0.04, 0.09, center=(sx_m, 1.08, 2.28)))

    chassis_mesh = b.add_mesh_primitives("ChassisMesh", [
        (md_ch_blk, m_chassis_blk),
        (md_ch_alu, m_brushed_alu),
        (md_ch_wht, m_white_paint),
        (md_ch_amb, m_amber_lamp),
    ])
    child_nodes.append(b.add_node("Chassis_Frame", mesh_idx=chassis_mesh))

    # Build left & right wheel meshes + spare wheel!
    wf_l, wt_l, wb_l = build_photoreal_wheel("+x")
    wf_r, wt_r, wb_r = build_photoreal_wheel("-x")
    mesh_wheel_l = b.add_mesh_primitives("WheelLeftMesh",  [(wf_l, m_wheel_face), (wt_l, m_tire_tread), (wb_l, m_chrome)])
    mesh_wheel_r = b.add_mesh_primitives("WheelRightMesh", [(wf_r, m_wheel_face), (wt_r, m_tire_tread), (wb_r, m_chrome)])

    for wname, wx, wz, is_left in [
        ("Wheel_FL", +1.01, +3.02, True),
        ("Wheel_FR", -1.01, +3.02, False),
        ("Wheel_ML", +1.01, -1.68, True),
        ("Wheel_MR", -1.01, -1.68, False),
        ("Wheel_RL", +1.01, -3.02, True),
        ("Wheel_RR", -1.01, -3.02, False),
    ]:
        child_nodes.append(b.add_node(wname, mesh_idx=(mesh_wheel_l if is_left else mesh_wheel_r), translation=(wx, 0.545, wz)))

    # Upright Spare Wheel behind cab on right side (-X, photo_1!)
    child_nodes.append(b.add_node("Wheel_Spare", mesh_idx=mesh_wheel_r, translation=(-0.92, 1.86, 1.90)))

    # -------------------------------------------------------------------------
    # B. SCULPTED KAMAZ-43118 CAB (Rounded corners, flared arches, real photo UVs!)
    # -------------------------------------------------------------------------
    md_cab_front  = MeshData()
    md_cab_bumper = MeshData()
    md_cab_side_l = MeshData()
    md_cab_side_r = MeshData()
    md_cab_white  = MeshData()
    md_cab_trim   = MeshData()
    md_cab_chrome = MeshData()
    md_cab_glass  = MeshData()
    md_cab_visor  = MeshData()
    md_cab_amber  = MeshData()

    # Exact KAMAZ-43118 Cab shell: z_rear = 2.26, z_front = 4.16 (lower) -> 3.92 (roofline), x = -1.16..+1.16
    zc_r = 2.26
    xc_h = 1.16
    fillet = 0.11  # Smooth rounded vertical front corners!

    # Vertical slices (y, z_front, v_front_uv, v_side_uv)
    cab_slices = [
        (0.96, 4.16, 1.00, 0.86),  # Bottom of grille / door sill
        (1.48, 4.18, 0.72, 0.61),  # Beltline crease
        (2.02, 4.15, 0.45, 0.36),  # Base of windshield
        (2.78, 3.92, 0.05, 0.03),  # Top of windshield
        (2.89, 3.84, 0.00, 0.00),  # Roof brow
    ]

    # 1) Front flat center panel (x = -(xc_h - fillet) .. +(xc_h - fillet)), mapped to cab_front!
    xf_in = xc_h - fillet
    u_in_l = 0.5 + (xf_in / xc_h) * 0.48
    u_in_r = 0.5 - (xf_in / xc_h) * 0.48
    for s_i in range(len(cab_slices) - 1):
        y0, zf0, vf0, _ = cab_slices[s_i]
        y1, zf1, vf1, _ = cab_slices[s_i + 1]
        # Notice looking at front (+Z): right truck side (-X) is LEFT of photo_2 (u=0)!
        # Wait: in photo_2, driver side (+X, left of truck) is on the RIGHT of photo_2 (x=774, u=1)!
        # Let's check photo_2: the yellow pass card on the passenger side (-X) is at x=480 (u=0.05, LEFT of photo_2)!
        # YES! So x = -xf_in maps to u = 0.04, and x = +xf_in maps to u = 0.96!
        n_f = (0.0, (zf0 - zf1), (y1 - y0))
        md_cab_front.add_smooth_quad(
            (-xf_in, y0, zf0), (+xf_in, y0, zf0), (+xf_in, y1, zf1), (-xf_in, y1, zf1),
            n_f, n_f, n_f, n_f,
            uv0=(0.045, vf0), uv1=(0.955, vf0), uv2=(0.955, vf1), uv3=(0.045, vf1)
        )

        # Smooth 4-segment rounded front corners connecting front face to left (+X) & right (-X) side walls!
        for side_sign, m_side in [(+1.0, md_cab_side_l), (-1.0, md_cab_side_r)]:
            for k in range(4):
                t0, t1 = k / 4.0, (k + 1) / 4.0
                a0, a1 = t0 * (math.pi * 0.5), t1 * (math.pi * 0.5)
                x00 = side_sign * (xf_in + fillet * math.sin(a0))
                z00 = (zf0 - fillet) + fillet * math.cos(a0)
                x01 = side_sign * (xf_in + fillet * math.sin(a1))
                z01 = (zf0 - fillet) + fillet * math.cos(a1)
                x10 = side_sign * (xf_in + fillet * math.sin(a0))
                z10 = (zf1 - fillet) + fillet * math.cos(a0)
                x11 = side_sign * (xf_in + fillet * math.sin(a1))
                z11 = (zf1 - fillet) + fillet * math.cos(a1)
                n0 = (side_sign * math.sin(a0), 0.08, math.cos(a0))
                n1 = (side_sign * math.sin(a1), 0.08, math.cos(a1))
                if side_sign > 0:
                    md_cab_white.add_smooth_quad((x00, y0, z00), (x01, y0, z01), (x11, y1, z11), (x10, y1, z10), n0, n1, n1, n0)
                else:
                    md_cab_white.add_smooth_quad((x01, y0, z01), (x00, y0, z00), (x10, y1, z10), (x11, y1, z11), n1, n0, n0, n1)

    # 2) Cab Left (+X) & Right (-X) Doors & Side Panels mapped to cab_side_l / cab_side_r!
    # In cab_side_l (1024x1024 from photo_6): u=0.05 is front of cab (z_front - fillet), u=0.98 is rear of cab (zc_r),
    # v=0.02 is roofline (y=2.89), v=0.86 is lower cab skirt (y=0.96)!
    for s_i in range(len(cab_slices) - 1):
        y0, zf0, _, vs0 = cab_slices[s_i]
        y1, zf1, _, vs1 = cab_slices[s_i + 1]
        z_f0 = zf0 - fillet
        z_f1 = zf1 - fillet
        u_f0 = 0.05 + (4.05 - z_f0) * 0.48
        u_f1 = 0.05 + (4.05 - z_f1) * 0.48
        # Left side (+X): front is left of photo_6 (u_f), rear is right of photo_6 (u=0.98)
        md_cab_side_l.add_quad(
            (+xc_h, y0, zc_r), (+xc_h, y0, z_f0), (+xc_h, y1, z_f1), (+xc_h, y1, zc_r),
            uv0=(0.98, vs0), uv1=(u_f0, vs0), uv2=(u_f1, vs1), uv3=(0.98, vs1), normal=(1, 0, 0)
        )
        # Right side (-X): cab_side_r is horizontally mirrored, so front is u = 1.0 - u_f, rear is u = 0.02!
        md_cab_side_r.add_quad(
            (-xc_h, y0, z_f0), (-xc_h, y0, zc_r), (-xc_h, y1, zc_r), (-xc_h, y1, z_f1),
            uv0=(1.0 - u_f0, vs0), uv1=(0.02, vs0), uv2=(0.02, vs1), uv3=(1.0 - u_f1, vs1), normal=(-1, 0, 0)
        )

    # Crowned Curved Cab Roof (y = 2.89 at sides curving smoothly up to 2.97 at x=0!) + Rear Cab Wall + Floor
    roof_segs = 8
    for k in range(roof_segs):
        t0, t1 = k / roof_segs, (k + 1) / roof_segs
        x0 = -xc_h + t0 * (2 * xc_h)
        x1 = -xc_h + t1 * (2 * xc_h)
        yr0 = 2.89 + 0.075 * math.sin(t0 * math.pi)
        yr1 = 2.89 + 0.075 * math.sin(t1 * math.pi)
        n0 = (-0.12 * math.cos(t0 * math.pi), 1.0, 0.0)
        n1 = (-0.12 * math.cos(t1 * math.pi), 1.0, 0.0)
        md_cab_white.add_smooth_quad((x0, yr0, 3.84), (x1, yr1, 3.84), (x1, yr1, zc_r), (x0, yr0, zc_r), n0, n1, n1, n0)
        # Rear cab wall strip
        md_cab_white.add_quad((x1, 0.96, zc_r), (x0, 0.96, zc_r), (x0, yr0, zc_r), (x1, yr1, zc_r), normal=(0, 0, -1))
    md_cab_trim.merge(make_box(2.28, 0.06, 1.84, center=(0.0, 0.96, 3.18)))

    # 3) Sculpted Flared Wheel Arches (расширители крыльев) & Front Boarding Footstep Wells!
    for sx_s in [-1.0, +1.0]:
        # Curved wheel arch over front wheel (centered at z=3.02, y=0.58, radius 0.64 -> 0.74)
        arch_segs = 14
        for k in range(arch_segs):
            a0 = math.radians(15 + k * (150.0 / arch_segs))
            a1 = math.radians(15 + (k + 1) * (150.0 / arch_segs))
            z_in0, y_in0 = 3.02 + 0.63 * math.cos(a0), 0.56 + 0.63 * math.sin(a0)
            z_in1, y_in1 = 3.02 + 0.63 * math.cos(a1), 0.56 + 0.63 * math.sin(a1)
            z_out0, y_out0 = 3.02 + 0.72 * math.cos(a0), 0.56 + 0.72 * math.sin(a0)
            z_out1, y_out1 = 3.02 + 0.72 * math.cos(a1), 0.56 + 0.72 * math.sin(a1)
            xi = sx_s * 0.96
            xo = sx_s * 1.23
            # Outer arch lip (matte black thermoplastic) & under-fender liner
            if sx_s > 0:
                md_cab_trim.add_quad((xo, y_in0, z_in0), (xo, y_in1, z_in1), (xo, y_out1, z_out1), (xo, y_out0, z_out0), normal=(1, 0, 0))
                md_cab_trim.add_quad((xi, y_in0, z_in0), (xo, y_in0, z_in0), (xo, y_in1, z_in1), (xi, y_in1, z_in1))
            else:
                md_cab_trim.add_quad((xo, y_in1, z_in1), (xo, y_in0, z_in0), (xo, y_out0, z_out0), (xo, y_out1, z_out1), normal=(-1, 0, 0))
                md_cab_trim.add_quad((xo, y_in0, z_in0), (xi, y_in0, z_in0), (xi, y_in1, z_in1), (xo, y_in1, z_in1))
        # Amber side turn indicator on fender arch
        md_cab_amber.merge(make_box(0.03, 0.038, 0.085, center=(sx_s * 1.235, 1.24, 3.22)))
        # White corner boarding step pocket + 2 perforated diamond-plate steps (photo_1: y=0.58 & y=0.80, z=3.86)
        md_cab_white.merge(make_beveled_box(0.22, 0.44, 0.44, bevel=0.025, center=(sx_s * 1.09, 0.74, 3.86)))
        for step_y in [0.56, 0.78]:
            md_cab_trim.merge(make_box(0.25, 0.028, 0.36, center=(sx_s * 1.11, step_y, 3.86)))

    # 4) Sculpted Steel Front Bumper (mapped to cab_bumper from photo_2!)
    zb_f = 4.29
    zb_r = 4.02
    xb_in = 1.04
    xb_out = 1.21
    yb0, yb1 = 0.52, 0.93
    # Center flat bumper face (u = 0.07..0.93)
    md_cab_bumper.add_quad(
        (-xb_in, yb0, zb_f), (+xb_in, yb0, zb_f), (+xb_in, yb1, zb_f), (-xb_in, yb1, zb_f),
        uv0=(0.07, 1.0), uv1=(0.93, 1.0), uv2=(0.93, 0.0), uv3=(0.07, 0.0), normal=(0, 0, 1)
    )
    # Angled wrap-around bumper wings (headlight corners!)
    md_cab_bumper.add_quad(
        (-xb_out, yb0, zb_r), (-xb_in, yb0, zb_f), (-xb_in, yb1, zb_f), (-xb_out, yb1, zb_r),
        uv0=(0.0, 1.0), uv1=(0.07, 1.0), uv2=(0.07, 0.0), uv3=(0.0, 0.0)
    )
    md_cab_bumper.add_quad(
        (+xb_in, yb0, zb_f), (+xb_out, yb0, zb_r), (+xb_out, yb1, zb_r), (+xb_in, yb1, zb_f),
        uv0=(0.93, 1.0), uv1=(1.0, 1.0), uv2=(1.0, 0.0), uv3=(0.93, 0.0)
    )
    # Top & bottom bumper ledges + 3D center tow pin jaw
    md_cab_white.add_quad((-xb_in, yb1, zb_f), (+xb_in, yb1, zb_f), (+xb_out, yb1, zb_r), (-xb_out, yb1, zb_r), normal=(0, 1, 0))
    md_cab_trim.merge(make_box(0.22, 0.12, 0.12, center=(0.0, 0.72, zb_f + 0.03)))

    # 5) 3D Sun Visor, 3 Amber Roof Lights, 2 Outer Spotlights, 3 Wipers & Mirrors!
    # Curved smoked acrylic sun visor across windshield brow
    vis_segs = 6
    for k in range(vis_segs):
        t0, t1 = k / vis_segs, (k + 1) / vis_segs
        vx0 = -1.16 + t0 * 2.32
        vx1 = -1.16 + t1 * 2.32
        vz0 = 4.02 + 0.11 * math.sin(t0 * math.pi)
        vz1 = 4.02 + 0.11 * math.sin(t1 * math.pi)
        md_cab_visor.add_quad((vx0, 2.67, vz0 + 0.07), (vx1, 2.67, vz1 + 0.07), (vx1, 2.86, vz1 - 0.06), (vx0, 2.86, vz0 - 0.06))
    for bx_v in [-0.85, -0.28, +0.28, +0.85]:
        md_cab_chrome.merge(make_cylinder(0.01, 0.01, 0.12, segs=8, axis="z", center=(bx_v, 2.79, 3.96)))

    # 3 Amber roof clearance lights (photo_2: center of roof brow!)
    for ax in [-0.19, 0.0, +0.19]:
        md_cab_amber.merge(make_beveled_box(0.065, 0.042, 0.085, bevel=0.012, center=(ax, 2.97, 3.78)))

    # 2 Round outer auxiliary spotlights on visor ends (photo_2!)
    for sx_sp in [-1.18, +1.18]:
        md_cab_trim.merge(make_cylinder(0.078, 0.068, 0.075, segs=18, axis="z", center=(sx_sp, 2.68, 4.00)))
        md_cab_chrome.merge(make_cylinder(0.068, 0.068, 0.015, segs=18, axis="z", center=(sx_sp, 2.68, 4.04)))

    # 3) Articulated Windshield Wipers + 2 Front Grab Handles
    for wx in [-0.56, 0.0, +0.56]:
        md_cab_trim.merge(make_box(0.36, 0.020, 0.022), offset=(wx + 0.12, 2.08, 4.17), rot_z_deg=-22)
        md_cab_trim.merge(make_box(0.40, 0.016, 0.018), offset=(wx + 0.24, 2.18, 4.16), rot_z_deg=6)
    for gx in [-0.58, +0.58]:
        md_cab_trim.merge(make_cylinder(0.014, 0.014, 0.42, segs=8, axis="x", center=(gx, 1.94, 4.19)))

    # Tubular Side Mirror Assemblies (C-brackets + main & wide-angle convex mirrors + chrome glass!)
    for sx_m in [-1.0, +1.0]:
        # Upper & lower horizontal arms + vertical mast
        md_cab_trim.merge(make_cylinder(0.015, 0.015, 0.26, segs=10, axis="x", center=(sx_m * 1.26, 2.66, 3.74)))
        md_cab_trim.merge(make_cylinder(0.015, 0.015, 0.26, segs=10, axis="x", center=(sx_m * 1.26, 2.00, 3.82)))
        md_cab_trim.merge(make_cylinder(0.016, 0.016, 0.68, segs=10, axis="y", center=(sx_m * 1.38, 2.33, 3.78)))
        # Main rectangular mirror housing + reflective face (-Z facing driver)
        md_cab_trim.merge(make_beveled_box(0.11, 0.36, 0.07, bevel=0.018, center=(sx_m * 1.38, 2.43, 3.78)))
        md_cab_chrome.merge(make_box(0.09, 0.32, 0.01, center=(sx_m * 1.38, 2.43, 3.744)))
        # Lower wide-angle mirror
        md_cab_trim.merge(make_beveled_box(0.11, 0.16, 0.065, bevel=0.016, center=(sx_m * 1.38, 2.13, 3.79)))
        md_cab_chrome.merge(make_box(0.09, 0.13, 0.01, center=(sx_m * 1.38, 2.13, 3.756)))
    # Curb look-down mirror above right passenger door (photo_1!)
    md_cab_trim.merge(make_cylinder(0.013, 0.013, 0.22, segs=8, axis="x", center=(-1.24, 2.74, 3.52)))
    md_cab_trim.merge(make_beveled_box(0.16, 0.09, 0.12, bevel=0.02, center=(-1.34, 2.72, 3.52)))

    cab_mesh = b.add_mesh_primitives("CabMesh", [
        (md_cab_front,  m_cab_front),
        (md_cab_bumper, m_cab_bumper),
        (md_cab_side_l, m_cab_side_l),
        (md_cab_side_r, m_cab_side_r),
        (md_cab_white,  m_white_paint),
        (md_cab_trim,   m_matte_plast),
        (md_cab_chrome, m_chrome),
        (md_cab_glass,  m_dark_glass),
        (md_cab_visor,  m_visor_smoke),
        (md_cab_amber,  m_amber_lamp),
    ])
    child_nodes.append(b.add_node("Cab_Body", mesh_idx=cab_mesh))

    # Realistic Two-Dome Cobalt-Blue Emergency Lightbar (СГУ, photo_12!)
    md_bcn_blue = MeshData()
    md_bcn_wht  = MeshData()
    md_bcn_chr  = MeshData()
    # Center white speaker block + left/right translucent blue aerodynamic domes with internal chrome reflectors!
    md_bcn_wht.merge(make_beveled_box(0.34, 0.105, 0.24, bevel=0.025, center=(0.0, 3.01, 3.24)))
    for sx_b in [-0.42, +0.42]:
        md_bcn_blue.merge(make_beveled_box(0.52, 0.105, 0.24, bevel=0.035, center=(sx_b, 3.01, 3.24)))
        md_bcn_chr.merge(make_box(0.38, 0.055, 0.12, center=(sx_b, 3.00, 3.24)))
        md_bcn_chr.merge(make_box(0.08, 0.05, 0.18, center=(sx_b, 2.94, 3.24)))
    beacon_mesh = b.add_mesh_primitives("BeaconMesh", [
        (md_bcn_blue, m_beacon_blue),
        (md_bcn_wht,  m_white_paint),
        (md_bcn_chr,  m_chrome),
    ])
    child_nodes.append(b.add_node("Beacon_Bar", mesh_idx=beacon_mesh))

    # -------------------------------------------------------------------------
    # C. PHOTOREALISTIC KUNG SHELTER, 3D INTERIORS, STAIRS & ROOF COMPLEX
    # -------------------------------------------------------------------------
    zk0, zk1 = -4.48, +1.36
    xk = 1.25
    yk0, yk_bev, yk_top = 1.16, 2.98, 3.26
    xk_top = 0.96

    md_kl    = MeshData()
    md_kr    = MeshData()
    md_krear = MeshData()
    md_kroof = MeshData()
    md_kwht  = MeshData()
    md_ktrim = MeshData()
    md_kalu  = MeshData()
    md_kint  = MeshData()
    md_ksr   = MeshData()
    md_kyel  = MeshData()
    md_ktail = MeshData()
    md_kamb  = MeshData()

    # Left Wall (+X) mapped 1:1 to kung_left (u=0 Front zk1 -> u=1 Rear zk0)
    md_kl.add_quad(
        (+xk, yk0, zk1), (+xk, yk0, zk0), (+xk, yk_bev, zk0), (+xk, yk_bev, zk1),
        uv0=(0.0, 1.0), uv1=(1.0, 1.0), uv2=(1.0, 0.0), uv3=(0.0, 0.0), normal=(1, 0, 0)
    )
    # Right Wall (-X) mapped 1:1 to kung_right (u=0 Rear zk0 -> u=1 Front zk1)
    md_kr.add_quad(
        (-xk, yk0, zk0), (-xk, yk0, zk1), (-xk, yk_bev, zk1), (-xk, yk_bev, zk0),
        uv0=(0.0, 1.0), uv1=(1.0, 1.0), uv2=(1.0, 0.0), uv3=(0.0, 0.0), normal=(-1, 0, 0)
    )
    # Chamfered upper bevel slopes (left & right) + Front bulkhead + Underfloor
    md_kwht.add_quad((+xk, yk_bev, zk1), (+xk, yk_bev, zk0), (+xk_top, yk_top, zk0), (+xk_top, yk_top, zk1), normal=(+0.7, +0.7, 0.0))
    md_kwht.add_quad((-xk, yk_bev, zk0), (-xk, yk_bev, zk1), (-xk_top, yk_top, zk1), (-xk_top, yk_top, zk0), normal=(-0.7, +0.7, 0.0))
    # Front KUNG bulkhead (+Z)
    md_kwht.add_quad((+xk, yk0, zk1), (-xk, yk0, zk1), (-xk, yk_bev, zk1), (+xk, yk_bev, zk1), normal=(0, 0, 1))
    md_kwht.add_quad((+xk, yk_bev, zk1), (-xk, yk_bev, zk1), (-xk_top, yk_top, zk1), (+xk_top, yk_top, zk1), normal=(0, 0, 1))
    # Rear KUNG wall (-Z) mapped to kung_rear!
    md_krear.add_quad(
        (-xk, yk0, zk0), (+xk, yk0, zk0), (+xk, yk_bev, zk0), (-xk, yk_bev, zk0),
        uv0=(0.0, 1.0), uv1=(1.0, 1.0), uv2=(1.0, 0.12), uv3=(0.0, 0.12), normal=(0, 0, -1)
    )
    md_krear.add_quad(
        (-xk, yk_bev, zk0), (+xk, yk_bev, zk0), (+xk_top, yk_top, zk0), (-xk_top, yk_top, zk0),
        uv0=(0.0, 0.12), uv1=(1.0, 0.12), uv2=(0.88, 0.0), uv3=(0.12, 0.0), normal=(0, 0, -1)
    )
    # Roof Deck mapped 1:1 to kung_roof (768x2048: u=0 Left +X -> u=1 Right -X; v=0 Front zk1 -> v=1 Rear zk0)!
    md_kroof.add_quad(
        (+xk_top, yk_top, zk1), (-xk_top, yk_top, zk1), (-xk_top, yk_top, zk0), (+xk_top, yk_top, zk0),
        uv0=(0.0, 0.0), uv1=(1.0, 0.0), uv2=(1.0, 1.0), uv3=(0.0, 1.0), normal=(0, 1, 0)
    )

    # 3D Extruded Rubber Window Bezels around all 4 KUNG windows for physical relief!
    def add_window_frame_3d(x_wall, sign_x, z_center, y_center, w_z=0.72, h_y=0.58):
        hz, hy = w_z * 0.5, h_y * 0.5
        t = 0.026
        md_ktrim.merge(make_box(0.022, t, w_z + 0.04, center=(x_wall + sign_x * 0.008, y_center + hy, z_center)))
        md_ktrim.merge(make_box(0.022, t, w_z + 0.04, center=(x_wall + sign_x * 0.008, y_center - hy, z_center)))
        md_ktrim.merge(make_box(0.022, h_y, t, center=(x_wall + sign_x * 0.008, y_center, z_center + hz)))
        md_ktrim.merge(make_box(0.022, h_y, t, center=(x_wall + sign_x * 0.008, y_center, z_center - hz)))

    # Left wall 3 windows (matching kung_left UV positions: u ≈ 0.195, 0.418, 0.598)
    for u_w in [0.195, 0.418, 0.598]:
        zw = zk1 - u_w * (zk1 - zk0)
        add_window_frame_3d(+xk, +1.0, zw, 2.58, w_z=0.72, h_y=0.58)
    # Right wall 1 window (u ≈ 0.302 from rear zk0)
    zw_r = zk0 + 0.302 * (zk1 - zk0)
    add_window_frame_3d(-xk, -1.0, zw_r, 2.58, w_z=0.72, h_y=0.58)

    # White Cylindrical Shelter/Antenna Roll (тубус) strapped to front-right of KUNG (photo_1!)
    md_kwht.merge(make_cylinder(0.21, 0.21, 1.08, segs=22, axis="x", center=(-0.46, 3.04, zk1 + 0.22)))
    for sx_st in [-0.78, -0.14]:
        md_ktrim.merge(make_cylinder(0.218, 0.218, 0.035, segs=22, axis="x", center=(sx_st, 3.04, zk1 + 0.22), caps=False))

    # 3D Recessed Side Communications Room (inside right doorway z = -1.38 .. +0.18)
    z_sd0, z_sd1 = -1.38, +0.18
    md_kint.add_quad((-0.78, 1.26, z_sd1), (-0.78, 1.26, z_sd0), (-0.78, 2.86, z_sd0), (-0.78, 2.86, z_sd1), normal=(-1, 0, 0))
    md_ktrim.merge(make_box(0.46, 0.04, 1.62, center=(-1.02, 1.25, -0.60)))  # Floor
    md_ktrim.merge(make_box(0.46, 0.04, 1.62, center=(-1.02, 2.88, -0.60)))  # Ceiling

    # 3D Recessed Rear Technical Bay + 2 Black Tactical Cable Drums (photo_9 & photo_10!)
    # Deep dark side walls & ceiling inside rear bay so exterior never sees floating rack edges!
    md_ktrim.merge(make_box(1.66, 0.05, 0.72, center=(0.0, 1.24, zk0 + 0.36)))
    md_ktrim.merge(make_box(1.66, 0.05, 0.72, center=(0.0, 2.86, zk0 + 0.36)))
    md_ktrim.merge(make_box(0.05, 1.62, 0.72, center=(-0.82, 2.05, zk0 + 0.36)))
    md_ktrim.merge(make_box(0.05, 1.62, 0.72, center=(+0.82, 2.05, zk0 + 0.36)))
    md_kint.add_quad((+0.78, 1.28, zk0 + 0.70), (-0.78, 1.28, zk0 + 0.70), (-0.78, 2.82, zk0 + 0.70), (+0.78, 2.82, zk0 + 0.70), normal=(0, 0, -1))
    # 2 Tactical cable reels / equipment cases sitting at rear bay threshold
    for cx_d in [-0.34, +0.34]:
        md_ktrim.merge(make_cylinder(0.24, 0.24, 0.22, segs=20, axis="x", center=(cx_d, 1.52, zk0 + 0.26)))
        md_kalu.merge(make_cylinder(0.08, 0.08, 0.25, segs=14, axis="x", center=(cx_d, 1.52, zk0 + 0.26)))

    # 3D Rear Multi-Chamber Taillights + Side Clearance Marker Lights
    for sx_t in [-0.94, +0.94]:
        md_ktrim.merge(make_beveled_box(0.30, 0.14, 0.05, bevel=0.015, center=(sx_t, 1.04, zk0 - 0.02)))
        md_ktail.merge(make_box(0.18, 0.10, 0.02, center=(sx_t, 1.04, zk0 - 0.042)))
        md_kamb.merge(make_box(0.08, 0.10, 0.02, center=(sx_t + (0.08 if sx_t > 0 else -0.08), 1.04, zk0 - 0.042)))

    # 3D Roof Complex: Tubular Grab Rails, 2 Beveled Roof Hatches, Whip Antennas, SUPER SKYRIDER Rack!
    for sx_r in [-0.91, +0.91]:
        md_kwht.merge(make_cylinder(0.022, 0.022, 5.56, segs=12, axis="z", center=(sx_r, yk_top + 0.14, -1.56)))
        for sz_p in np.linspace(zk0 + 0.20, zk1 - 0.20, 8):
            md_kwht.merge(make_cylinder(0.018, 0.018, 0.14, segs=10, axis="y", center=(sx_r, yk_top + 0.07, float(sz_p))))
    md_kwht.merge(make_cylinder(0.022, 0.022, 1.82, segs=12, axis="x", center=(0.0, yk_top + 0.14, zk0 + 0.16)))

    # 2 Beveled White Roof Hatches (photo_12: front z=+0.56, rear z=-1.82)
    for hz_h in [+0.56, -1.82]:
        md_kwht.merge(make_beveled_box(0.66, 0.09, 0.66, bevel=0.03, center=(0.0, yk_top + 0.045, hz_h)))
        md_ktrim.merge(make_box(0.68, 0.02, 0.68, center=(0.0, yk_top + 0.01, hz_h)))

    # Black Perforated SUPER SKYRIDER Roof Expedition Rack (z = -0.62, photo_12!)
    rack_z = -0.62
    for sx_rk in [-0.74, +0.74]:
        md_ktrim.merge(make_beveled_box(0.065, 0.07, 1.86, bevel=0.015, center=(sx_rk, yk_top + 0.055, rack_z)))
        md_ktrim.merge(make_cylinder(0.020, 0.020, 1.86, segs=10, axis="z", center=(sx_rk, yk_top + 0.145, rack_z)))
    for sz_s in np.linspace(rack_z - 0.86, rack_z + 0.86, 11):
        md_ktrim.merge(make_box(1.44, 0.032, 0.105, center=(0.0, yk_top + 0.05, float(sz_s))))
    for sz_end in [rack_z - 0.92, rack_z + 0.92]:
        md_ktrim.merge(make_cylinder(0.020, 0.020, 1.48, segs=10, axis="x", center=(0.0, yk_top + 0.145, sz_end)))
    md_kyel.merge(make_box(0.09, 0.045, 0.26, center=(+0.67, yk_top + 0.08, rack_z + 0.70)))
    md_ksr.add_quad((-0.62, yk_top + 0.055, rack_z + 0.94), (+0.62, yk_top + 0.055, rack_z + 0.94), (+0.62, yk_top + 0.21, rack_z + 0.88), (-0.62, yk_top + 0.21, rack_z + 0.88))

    # Roof HF/VHF Whip Antennas on spring mounts
    for wx, wz, wh in [(-0.82, 0.88, 1.05), (-0.82, -2.55, 1.15), (+0.82, -2.55, 0.95)]:
        md_ktrim.merge(make_cylinder(0.032, 0.026, 0.11, segs=10, axis="y", center=(wx, yk_top + 0.055, wz)))
        md_kalu.merge(make_cylinder(0.009, 0.003, wh, segs=8, axis="y", center=(wx, yk_top + 0.11 + wh * 0.5, wz)))

    kung_mesh = b.add_mesh_primitives("KungBodyMesh", [
        (md_kl,    m_kung_l),
        (md_kr,    m_kung_r),
        (md_krear, m_kung_rear),
        (md_kroof, m_kung_roof),
        (md_kwht,  m_white_paint),
        (md_ktrim, m_matte_plast),
        (md_kalu,  m_brushed_alu),
        (md_kint,  m_interior),
        (md_ksr,   m_skyrider),
        (md_kyel,  m_safety_yel),
        (md_ktail, m_tail_red),
        (md_kamb,  m_amber_lamp),
    ])
    child_nodes.append(b.add_node("Kung_Body", mesh_idx=kung_mesh))

    # -------------------------------------------------------------------------
    # D. ANIMATED KUNG DOORS, PERFORATED STAIRS & REAR GRATED RAMP
    # -------------------------------------------------------------------------
    def build_kung_side_door(z_sign):
        """z_sign=-1 hinges at front (+z -> opens toward +z), z_sign=+1 hinges at rear (-z -> opens toward -z)."""
        dw = 0.76
        zc = z_sign * (dw * 0.5)
        md_w = make_beveled_box(0.055, 1.56, dw, bevel=0.015, center=(0.0, 0.0, zc))
        md_in = make_box(0.015, 1.44, dw - 0.08, center=(+0.025, 0.0, zc))
        md_tr = make_box(0.068, 0.48, dw - 0.18, center=(0.0, 0.34, zc))  # Tinted door window + gasket
        md_or = make_box(0.062, 0.16, dw - 0.02, center=(-0.002, -0.34, zc))
        md_bl = make_box(0.065, 0.055, dw - 0.02, center=(-0.003, -0.34, zc))
        # Compression T-latch handle
        md_tr.merge(make_box(0.04, 0.12, 0.04, center=(-0.035, -0.08, zc + z_sign * 0.24)))
        return b.add_mesh_primitives("SideDoorMesh", [
            (md_w,  m_white_paint),
            (md_in, m_inner_door),
            (md_tr, m_dark_glass),
            (md_or, m_orange_acc),
            (md_bl, m_blue_acc),
        ])

    child_nodes.append(b.add_node("SideDoor_F", mesh_idx=build_kung_side_door(-1.0), translation=(-xk - 0.02, 2.06, +0.16)))
    child_nodes.append(b.add_node("SideDoor_R", mesh_idx=build_kung_side_door(+1.0), translation=(-xk - 0.02, 2.06, -1.36)))

    # Twin Black Perforated Steel Side Stairs + Handrails (SideStairs, pivot at (-xk, 1.20, -0.60))
    md_st_blk = MeshData()
    md_st_alu = MeshData()
    md_st_org = MeshData()
    md_st_blk.merge(make_box(0.68, 0.045, 1.58, center=(-0.34, 0.0, 0.0)))
    md_st_alu.merge(make_box(0.64, 0.050, 1.52, center=(-0.34, 0.004, 0.0)))
    for z_flight in [+0.40, -0.40]:
        for s_off in [-0.24, +0.24]:
            # Single clean continuous slanted stringer beam from (-0.64, -0.02) down to (-1.46, -1.16)
            md_st_blk.merge(make_box(1.42, 0.048, 0.032), offset=(-1.05, -0.59, z_flight + s_off), rot_z_deg=54.2)
        for step_i in range(5):
            t = (step_i + 1) / 5.6
            md_st_blk.merge(make_box(0.24, 0.028, 0.48, center=(-0.62 - t * 0.82, -t * 1.12, z_flight)))
            md_st_alu.merge(make_box(0.20, 0.032, 0.44, center=(-0.62 - t * 0.82, -t * 1.12 + 0.003, z_flight)))
        for hr_s in [-0.25, +0.25]:
            md_st_blk.merge(make_cylinder(0.016, 0.016, 0.88, segs=8, axis="y", center=(-0.64, 0.44, z_flight + hr_s)))
            md_st_blk.merge(make_cylinder(0.016, 0.016, 0.88, segs=8, axis="y", center=(-1.42, -0.66, z_flight + hr_s)))
            # Continuous slanted matte-black steel handrail bar connecting upper & lower posts (photo_1 & photo_10!)
            md_st_blk.merge(make_box(1.36, 0.026, 0.026), offset=(-1.03, +0.33, z_flight + hr_s), rot_z_deg=54.5)
    stairs_mesh = b.add_mesh_primitives("SideStairsMesh", [
        (md_st_blk, m_chassis_blk),
        (md_st_alu, m_brushed_alu),
    ])
    child_nodes.append(b.add_node("SideStairs", mesh_idx=stairs_mesh, translation=(-xk, 1.20, -0.60)))

    # Double Rear Compartment Doors (photo_10: vertical orange-blue-orange stripe on outer center edges + grey inner skin!)
    def build_rear_door(x_sign):
        """x_sign=-1 is Left (+X) hinge -> center is toward -X; x_sign=+1 is Right (-X) hinge -> center is toward +X."""
        dw = 0.82
        xc = x_sign * (dw * 0.5)
        md_w  = make_beveled_box(dw, 1.54, 0.055, bevel=0.015, center=(xc, 0.0, 0.0))
        md_in = make_box(dw - 0.08, 1.42, 0.016, center=(xc, 0.0, +0.024))
        md_tr = make_box(dw + 0.01, 1.55, 0.025, center=(xc, 0.0, +0.01))  # Perimeter rubber gasket
        # Vertical stripe near center meeting edge (photo_10!) + horizontal lower stripe
        edge_x = x_sign * (dw - 0.10)
        md_or = make_box(0.14, 1.50, 0.060, center=(edge_x, 0.0, -0.002))
        md_bl = make_box(0.055, 1.50, 0.063, center=(edge_x, 0.0, -0.003))
        md_or.merge(make_box(dw - 0.03, 0.14, 0.060, center=(xc, -0.34, -0.002)))
        md_bl.merge(make_box(dw - 0.03, 0.05, 0.063, center=(xc, -0.34, -0.003)))
        # Black inner T-latches (photo_10!)
        for ly in [-0.28, +0.22]:
            md_tr.merge(make_box(0.14, 0.035, 0.04, center=(x_sign * (dw * 0.68), ly, +0.04)))
        return b.add_mesh_primitives("RearDoorMesh", [
            (md_w,  m_white_paint),
            (md_in, m_inner_door),
            (md_tr, m_chassis_blk),
            (md_or, m_orange_acc),
            (md_bl, m_blue_acc),
        ])

    child_nodes.append(b.add_node("RearDoor_L", mesh_idx=build_rear_door(-1.0), translation=(+0.84, 2.02, zk0 - 0.025)))
    child_nodes.append(b.add_node("RearDoor_R", mesh_idx=build_rear_door(+1.0), translation=(-0.84, 2.02, zk0 - 0.025)))

    # Rear Grated Fold-Down Platform + Slanted Tubular Access Ladder (photo_10!)
    md_rp_blk = MeshData()
    md_rp_alu = MeshData()
    md_rp_blk.merge(make_box(1.74, 0.042, 0.88, center=(0.0, 0.0, -0.44)))
    for gz in np.linspace(-0.08, -0.82, 9):
        md_rp_alu.merge(make_box(1.66, 0.046, 0.035, center=(0.0, 0.002, float(gz))))
    # Slanted ladder on right-center of rear ramp (photo_10!)
    for lx in [-0.52, -0.08]:
        md_rp_blk.merge(make_box(0.035, 1.34, 0.035), offset=(lx, -0.58, -1.18), rot_x_deg=29.0)
    for step_i in range(5):
        t = (step_i + 1) / 5.5
        md_rp_blk.merge(make_cylinder(0.016, 0.016, 0.44, segs=10, axis="x", center=(-0.30, -t * 1.14, -0.86 - t * 0.62)))
    rear_plat_mesh = b.add_mesh_primitives("RearPlatMesh", [(md_rp_blk, m_chassis_blk), (md_rp_alu, m_brushed_alu)])
    child_nodes.append(b.add_node("RearPlatform", mesh_idx=rear_plat_mesh, translation=(0.0, 1.20, zk0 - 0.02)))

    # -------------------------------------------------------------------------
    # E. TELESCOPIC PNEUMATIC MAST + COILED RF CABLE + PTZ DOME CAMERA (photo_6!)
    # -------------------------------------------------------------------------
    mx, mz = +1.18, +1.52
    md_mb_wht = make_cylinder(0.072, 0.072, 2.28, segs=20, axis="y", center=(0.0, 1.14, 0.0))
    md_mb_blk = make_beveled_box(0.24, 0.34, 0.24, bevel=0.025, center=(-0.10, 0.22, 0.0))
    for cy_c in [0.65, 1.45, 2.18]:
        md_mb_blk.merge(make_cylinder(0.086, 0.086, 0.07, segs=20, axis="y", center=(0.0, cy_c, 0.0)))
        md_mb_wht.merge(make_box(0.22, 0.06, 0.08, center=(-0.10, cy_c, 0.0)))
    # Helical coiled black RF cable around mast base!
    for k in range(28):
        ang = k * 0.65
        ky = 0.45 + k * 0.062
        md_mb_blk.merge(make_box(0.022, 0.026, 0.035, center=(0.105 * math.cos(ang), ky, 0.105 * math.sin(ang))))
    mast_base_mesh = b.add_mesh_primitives("MastBaseMesh", [(md_mb_wht, m_white_paint), (md_mb_blk, m_chassis_blk)])

    md_ms1_alu = make_cylinder(0.054, 0.054, 1.55, segs=18, axis="y", center=(0.0, 0.775, 0.0))
    md_ms1_blk = make_cylinder(0.064, 0.064, 0.06, segs=18, axis="y", center=(0.0, 1.52, 0.0))
    mast_s1_mesh = b.add_mesh_primitives("MastS1Mesh", [(md_ms1_alu, m_brushed_alu), (md_ms1_blk, m_chassis_blk)])

    md_ms2_alu = make_cylinder(0.038, 0.038, 1.45, segs=16, axis="y", center=(0.0, 0.725, 0.0))
    md_ms2_wht = MeshData()
    md_ms2_gls = MeshData()
    # Crossbar + gooseneck PTZ camera housing + smoked optical dome + 2 VHF dipole whips!
    md_ms2_alu.merge(make_box(0.46, 0.032, 0.032, center=(0.0, 1.38, 0.0)))
    md_ms2_wht.merge(make_cylinder(0.022, 0.022, 0.20, segs=12, axis="z", center=(0.0, 1.24, +0.11)))
    md_ms2_wht.merge(make_cylinder(0.095, 0.105, 0.14, segs=20, axis="y", center=(0.0, 1.18, +0.20)))
    md_ms2_gls.merge(make_cylinder(0.082, 0.040, 0.08, segs=20, axis="y", center=(0.0, 1.08, +0.20)))
    for dx in [-0.21, +0.21]:
        md_ms2_alu.merge(make_cylinder(0.008, 0.004, 0.68, segs=8, axis="y", center=(dx, 1.72, 0.0)))
    mast_s2_mesh = b.add_mesh_primitives("MastS2Mesh", [(md_ms2_alu, m_brushed_alu), (md_ms2_wht, m_white_paint), (md_ms2_gls, m_dark_glass)])

    n_ms2 = b.add_node("Mast_Stage2", mesh_idx=mast_s2_mesh, translation=(0.0, 1.25, 0.0))
    n_ms1 = b.add_node("Mast_Stage1", mesh_idx=mast_s1_mesh, translation=(0.0, 2.05, 0.0), children=[n_ms2])
    n_mb  = b.add_node("Mast_Base",   mesh_idx=mast_base_mesh, translation=(mx, 0.96, mz), children=[n_ms1])
    child_nodes.append(n_mb)

    # -------------------------------------------------------------------------
    # F. CURVED PARABOLIC MOTORIZED VSAT iNetVu SATELLITE DISH (photo_11 & photo_12!)
    # -------------------------------------------------------------------------
    vx, vy, vz = 0.0, yk_top, -3.56
    md_vaz_blk = make_beveled_box(0.56, 0.14, 0.72, bevel=0.03, center=(0.0, 0.07, 0.12))
    md_vaz_blk.merge(make_cylinder(0.24, 0.26, 0.15, segs=24, axis="y", center=(0.0, 0.16, 0.0)))
    md_vaz_wht = make_beveled_box(0.22, 0.11, 0.24, bevel=0.02, center=(-0.38, 0.06, 0.26))  # Weatherproof roof junction box
    vsat_az_mesh = b.add_mesh_primitives("VsatAzMesh", [(md_vaz_blk, m_chassis_blk), (md_vaz_wht, m_white_paint)])

    # True 32-segment Curved Parabolic Dish Platter + Sculpted Motorized Feedhorn Boom!
    md_vel_dish = MeshData()
    md_vel_blk  = MeshData()
    md_vel_alu  = MeshData()

    dish_r = 0.68
    d_segs = 32
    d_rings = [0.0, 0.22, 0.45, 0.68]
    dish_cy, dish_cz = 0.08, -0.18
    for r_i in range(len(d_rings) - 1):
        r0, r1 = d_rings[r_i], d_rings[r_i + 1]
        # Concave parabolic dish surface (edges curve slightly UP toward feedhorn!)
        h0 = 0.02 + 0.11 * ((r0 / dish_r) ** 2)
        h1 = 0.02 + 0.11 * ((r1 / dish_r) ** 2)
        for i in range(d_segs):
            a0 = 2.0 * math.pi * i / d_segs
            a1 = 2.0 * math.pi * (i + 1) / d_segs
            c0, s0 = math.cos(a0), math.sin(a0)
            c1, s1 = math.cos(a1), math.sin(a1)
            p00 = (r0 * c0, dish_cy + h0, dish_cz + r0 * s0)
            p10 = (r0 * c1, dish_cy + h0, dish_cz + r0 * s1)
            p11 = (r1 * c1, dish_cy + h1, dish_cz + r1 * s1)
            p01 = (r1 * c0, dish_cy + h1, dish_cz + r1 * s0)
            n0 = (-0.25 * (r0 / dish_r) * c0, 1.0, -0.25 * (r0 / dish_r) * s0)
            n1 = (-0.25 * (r1 / dish_r) * c1, 1.0, -0.25 * (r1 / dish_r) * s1)
            uv00 = (0.5 - (r0 / dish_r) * 0.48 * c0, 0.5 - (r0 / dish_r) * 0.48 * s0)
            uv10 = (0.5 - (r0 / dish_r) * 0.48 * c1, 0.5 - (r0 / dish_r) * 0.48 * s1)
            uv11 = (0.5 - (r1 / dish_r) * 0.48 * c1, 0.5 - (r1 / dish_r) * 0.48 * s1)
            uv01 = (0.5 - (r1 / dish_r) * 0.48 * c0, 0.5 - (r1 / dish_r) * 0.48 * s0)
            if r0 < 1e-5:
                md_vel_dish.add_tri(p00, p11, p01, uv0=(0.5, 0.5), uv1=uv11, uv2=uv01, normal=(0, 1, 0))
            else:
                md_vel_dish.add_smooth_quad(p00, p10, p11, p01, n0, n0, n1, n1, uv00, uv10, uv11, uv01)
            # Underside backing ribs of dish
            md_vel_blk.add_quad(
                (r0 * c0, dish_cy + h0 - 0.025, dish_cz + r0 * s0),
                (r1 * c0, dish_cy + h1 - 0.025, dish_cz + r1 * s0),
                (r1 * c1, dish_cy + h1 - 0.025, dish_cz + r1 * s1),
                (r0 * c1, dish_cy + h0 - 0.025, dish_cz + r0 * s1),
                normal=(0, -1, 0)
            )

    # Sculpted aerodynamic iNetVu feedhorn arm & LNB transceiver (photo_11 & photo_12!)
    md_vel_blk.merge(make_beveled_box(0.22, 0.16, 0.34, bevel=0.035, center=(0.0, 0.12, +0.28)))
    # Curved boom reaching over dish focal point
    md_vel_blk.merge(make_beveled_box(0.15, 0.09, 0.68, bevel=0.03), offset=(0.0, 0.36, -0.02), rot_x_deg=-26)
    md_vel_alu.merge(make_cylinder(0.048, 0.038, 0.16, segs=16, axis="y", center=(0.0, 0.50, -0.28)),)
    vsat_el_mesh = b.add_mesh_primitives("VsatElMesh", [
        (md_vel_dish, m_vsat_dish),
        (md_vel_blk,  m_chassis_blk),
        (md_vel_alu,  m_brushed_alu),
    ])

    n_vel = b.add_node("VSAT_Elevation", mesh_idx=vsat_el_mesh, translation=(0.0, 0.22, 0.0), rotation_quat=(-0.29237, 0.0, 0.0, 0.95630))
    n_vaz = b.add_node("VSAT_Azimuth",   mesh_idx=vsat_az_mesh, translation=(vx, vy, vz), children=[n_vel])
    child_nodes.append(n_vaz)

    root_idx = b.add_node("Kamaz_Comms_Hub", children=child_nodes)
    size_bytes = b.export_glb(out_path, [root_idx])
    total_tris = sum(
        sum(b.accessors[p["indices"]]["count"] // 3 for p in m["primitives"])
        for m in b.meshes
    )
    print(f"Generated Photorealistic GLB: {out_path} ({size_bytes/1024/1024:.2f} MB, {len(b.nodes)} nodes, {total_tris} triangles)")


if __name__ == "__main__":
    build_kamaz_comms_hub_glb("/home/user/ar-comms-hub/public/models/kamaz_comms_hub.glb")

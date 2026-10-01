#!/usr/bin/env python3
"""
Export the Checkpoint 1 block model as CAD: one STEP + one STL per part, plus a
STEP assembly (part names + colours preserved) that opens in SolidWorks / Onshape / Fusion.

Setup:   pip install cadquery
Run:     python export_cad.py                      # frame-3 pose
         python export_cad.py --pose f4           # or f1, f3, f4
         python export_cad.py --yaw 0 --u 131 --f 250 --out cad_export   # -> exports/cad_export/

Coordinates (mm): origin = robot base axis on the plate's top surface,
X = right in side-A frames, Y = away from camera, Z = up.
Every part is modelled in this shared frame, so the assembly needs no mates to
sit correctly -- fix one part and the rest line up.

All numbers come from the averaged annotations. Parts flagged ASSUMED in the
wireframe viewer (plate size/holes, actuator lengths, depth stacking) are
placeholders; edit the D dict below and re-run.
"""
import argparse, math
from pathlib import Path
import cadquery as cq

D = dict(
    plateW=200.0, plateT=12.7, robotX=125.0,             # plate: ASSUMED nominal
    holeD=6.0, cbHole=6.6, cbD=10.3, cbDepth=6.5,        # hole dia / cbore depth: ASSUMED; cbD measured
    puckD=38.2, puckH=16.2, flangeT=4.6, gap=1.2,
    basePlateT=5.7, turretH=33.7, turretWx=61.6, turretWy=39.8,
    yawD=30.4, yawH=21.1, bracketH=90.2, bracketT=4.7, shoulderZ=94.8,
    ringD=62.4, hubOD=37.6, motorL=40.1, hubL=18.5,       # motorL/hubL: single measurement
    linkT=4.7, linkW=27.1, L1=99.4, L2=103.8,
    slotUp=65.7, slotFore=61.7, slotForeOff=22.0, slotW=5.1, tipD=12.9,
)
POSES = {"f3": (0, 131, 250), "f4": (0, 65, 154), "f1": (0, 125, 199), "f2": (90, 90, 270)}


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0, centered=False).translate((x0, y0, z0))

def cylz(r, z0, z1, x=0.0, y=0.0):
    return cq.Workplane("XY", origin=(x, y, z0)).circle(r).extrude(z1 - z0)

def cyly(r, y0, y1, cx, cz):
    """Cylinder with its axis along Y, spanning y0 < y < y1, centred at (cx, cz)."""
    return cq.Workplane("XZ", origin=(0, y1, 0)).center(cx, cz).circle(r).extrude(y1 - y0)

def link(O, th_deg, L, w, y0, y1, slot_start, slot_len, slot_w, tip_d=None):
    """Flat link plate in the X-Z plane: joint centre at O, running L along angle th (CCW from +X)."""
    t = y1 - y0
    wp = lambda: cq.Workplane("XZ", origin=(0, t, 0))
    body = wp().center(L / 2, 0).slot2D(L + w, w, 0).extrude(t)
    body = body.cut(wp().center(slot_start + slot_len / 2, 0).slot2D(slot_len, slot_w, 0).extrude(t))
    if tip_d:
        body = body.cut(wp().center(L, 0).circle(tip_d / 2).extrude(t))
    body = body.translate((0, y0, 0)).rotate((0, 0, 0), (0, 1, 0), -th_deg)
    return body.translate((O[0], 0, O[2]))


def build(yaw, u, f):
    d = D
    z_fl0, z_fl1 = d["puckH"], d["puckH"] + d["flangeT"]
    z_t0 = z_fl1 + d["gap"]
    z_top = z_t0 + d["basePlateT"] + d["turretH"]
    hw, hd = d["turretWx"] / 2, d["turretWy"] / 2
    # depth stack (Y), camera side is -Y
    yB1 = hd; yB0 = yB1 - d["bracketT"]
    yM1 = yB0; yM0 = yM1 - d["motorL"]
    yU1 = yM0; yU0 = yU1 - d["linkT"]
    yH1 = yU0; yH0 = yH1 - d["hubL"]
    yF1 = yH0; yF0 = yF1 - d["linkT"]
    R = d["ringD"] / 2
    S = (0.0, 0.0, d["shoulderZ"])
    E = (d["L1"] * math.cos(math.radians(u)), 0.0, S[2] + d["L1"] * math.sin(math.radians(u)))

    parts = {}  # name -> (solid, rotates_with_yaw, (r,g,b))
    # plate with 8x8 grid holes and 4 counterbores
    plate = box(-d["robotX"], d["plateW"] - d["robotX"], -d["plateW"] / 2, d["plateW"] / 2, -d["plateT"], 0)
    pts = [(-d["robotX"] + 12.5 + 25 * i, -d["plateW"] / 2 + 12.5 + 25 * j) for i in range(8) for j in range(8)]
    plate = plate.cut(cq.Workplane("XY", origin=(0, 0, -d["plateT"])).pushPoints(pts).circle(d["holeD"] / 2).extrude(d["plateT"]))
    cb = [(-d["robotX"] + a, b) for a in (25, 175) for b in (-75, 75)]
    plate = plate.cut(cq.Workplane("XY", origin=(0, 0, -d["plateT"])).pushPoints(cb).circle(d["cbHole"] / 2).extrude(d["plateT"]))
    plate = plate.cut(cq.Workplane("XY", origin=(0, 0, -d["cbDepth"])).pushPoints(cb).circle(d["cbD"] / 2).extrude(d["cbDepth"]))
    parts["01_optical_plate"] = (plate, False, (0.25, 0.25, 0.28))
    parts["02_base_puck"] = (cylz(d["puckD"] / 2, 0, d["puckH"]), False, (0.6, 0.62, 0.66))
    parts["03_yaw_flange"] = (cylz(hw, z_fl0, z_fl1), True, (0.7, 0.72, 0.75))
    turret = box(-hw, hw, -hd, yB0, z_t0, z_top)
    turret = turret.cut(cylz(d["yawD"] / 2, z_top - d["yawH"], z_top))      # pocket for the yaw actuator
    parts["04_turret"] = (turret, True, (0.75, 0.77, 0.8))
    parts["05_yaw_actuator"] = (cylz(d["yawD"] / 2, z_top - d["yawH"], z_top), True, (0.9, 0.9, 0.92))
    parts["06_shoulder_bracket"] = (box(-hw, hw, yB0, yB1, z_fl1 + 0.8, z_fl1 + 0.8 + d["bracketH"]), True, (0.75, 0.77, 0.8))
    parts["07_shoulder_actuator"] = (cyly(R, yM0, yM1, S[0], S[2]), True, (0.9, 0.9, 0.92))
    parts["08_upper_arm"] = (link(S, u, d["L1"], d["linkW"], yU0, yU1, (d["L1"] - d["slotUp"]) / 2, d["slotUp"], d["slotW"]), True, (0.72, 0.75, 0.78))
    parts["09_elbow_actuator"] = (cyly(R, yM0, yM1, E[0], E[2]), True, (0.9, 0.9, 0.92))
    parts["10_elbow_hub"] = (cyly(d["hubOD"] / 2, yH0, yH1, E[0], E[2]), True, (0.6, 0.62, 0.66))
    parts["11_forearm"] = (link(E, f, d["L2"], d["linkW"], yF0, yF1, d["slotForeOff"], d["slotFore"], d["slotW"], d["tipD"]), True, (0.72, 0.75, 0.78))

    out = {}
    for name, (solid, rot, col) in parts.items():
        if rot and yaw:
            solid = solid.rotate((0, 0, 0), (0, 0, 1), yaw)
        out[name] = (solid, col)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=POSES, default="f3")
    ap.add_argument("--yaw", type=float); ap.add_argument("--u", type=float); ap.add_argument("--f", type=float)
    ap.add_argument("--out", default="cad_export")
    a = ap.parse_args()
    yaw, u, f = POSES[a.pose]
    yaw = a.yaw if a.yaw is not None else yaw
    u = a.u if a.u is not None else u
    f = a.f if a.f is not None else f
    print(f"pose: yaw {yaw}  upper {u}  forearm {f}")

    out = Path("exports") / a.out          # relative to where you run the script
    (out / "step").mkdir(parents=True, exist_ok=True)
    (out / "stl").mkdir(parents=True, exist_ok=True)
    parts = build(yaw, u, f)
    asm = cq.Assembly(name="manipulator_block_model")
    for name, (solid, col) in parts.items():
        cq.exporters.export(solid, str(out / "step" / f"{name}.step"))
        cq.exporters.export(solid, str(out / "stl" / f"{name}.stl"), tolerance=0.05, angularTolerance=0.1)
        asm.add(solid, name=name, color=cq.Color(*col))
        bb = solid.val().BoundingBox()
        print(f"  {name:22s} x[{bb.xmin:7.1f},{bb.xmax:7.1f}] y[{bb.ymin:7.1f},{bb.ymax:7.1f}] z[{bb.zmin:7.1f},{bb.zmax:7.1f}]")
    try:
        asm.export(str(out / "manipulator_assembly.step"))
    except AttributeError:
        asm.save(str(out / "manipulator_assembly.step"))
    print(f"\nwrote {out}/manipulator_assembly.step, {len(parts)} part STEPs, {len(parts)} STLs")

if __name__ == "__main__":
    main()
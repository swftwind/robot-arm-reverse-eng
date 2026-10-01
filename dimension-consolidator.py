import json, math, sys, statistics as st
"""Usage: python consolidate_dimensions.py [annotations.json]
Re-measures every annotation with one shared scale and prints per-feature averages.
The feature->click mapping below is specific to the four frames used so far."""
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'annotations.json'))
keys = sorted(d)            # t=5.37, 14.25, 21.72, 36.20 -> frames 1..4
F = {i+1: d[k] for i, k in enumerate(keys)}

def find(fr, p1, p2):
    for e in F[fr]:
        if 'pts' in e:
            a, b = tuple(e['pts'][0]), tuple(e['pts'][1])
            if {a, b} == {tuple(p1), tuple(p2)}:
                return math.dist(a, b)
    raise KeyError((fr, p1, p2))

# unified scale from every 25 mm reference we have
refs = [e['px'] for fr in F for e in F[fr] if e['type'] == 'scale']
refs.append(find(4, (921,950), (841,950)))        # extra 25 mm check on frame 4
S = st.mean(refs) / 25
print(f"reference px: {[round(r,1) for r in refs]}  -> unified scale {S:.3f} px/mm (sd {st.pstdev(refs)/st.mean(refs)*100:.1f}%)")

feat = {
'puck_h':   [(1,(980,888),(980,934)),(2,(1044,888),(1044,940)),(4,(1036,885),(1036,942))],
'puck_d':   [(1,(1101,912),(979,913))],
'flange_t': [(1,(1051,872),(1051,887)),(2,(1071,874),(1071,887)),(4,(1006,873),(1006,889))],
'plate_t':  [(2,(1058,852),(1057,868)),(3,(1014,851),(1016,870)),(4,(996,849),(996,868))],
'turret_h': [(2,(1087,744),(1086,851)),(3,(969,741),(971,850)),(4,(968,742),(971,849))],
'turret_wx':[(3,(944,870),(1145,872)),(4,(945,871),(1137,871))],
'turret_wy':[(2,(1098,697),(971,695))],
'yaw_d':    [(4,(982,811),(1079,812))],
'yaw_h':    [(1,(1102,742),(1101,809)),(4,(1080,812),(1080,744))],
'bracket_h':[(3,(1146,589),(1148,879)),(4,(1140,871),(1145,585))],
'ring_r':   [(1,(1076,640),(972,640)),(3,(948,649),(1038,637)),(1,(899,376),(830,298)),(4,(1156,346),(1244,392))],
'hub_r':    [(1,(1078,638),(1103,672)),(3,(1040,636),(1039,680))],
'pcd_r':    [(1,(1076,638),(1044,690)),(4,(1156,348),(1184,296))],
'link_t':   [(2,(1063,452),(1078,452)),(2,(1067,305),(1082,305))],
'link_w':   [(3,(944,594),(1009,537))],
'slot_up':  [(1,(925,416),(1049,598)),(3,(866,440),(999,592)),(4,(1042,585),(1130,398))],
'slot_fore':[(1,(632,468),(820,397)),(4,(930,233),(1104,317))],
'slot_w':   [(1,(946,461),(958,451)),(1,(704,432),(710,447)),(4,(963,240),(955,255))],
'L1_upper': [(1,(899,378),(1078,637))],
'L2_fore':  [(1,(898,377),(586,486))],
'tip_hole': [(1,(584,487),(597,526)),(4,(877,207),(841,187))],
'bolt_head':[(1,(897,377),(909,364)),(1,(1079,637),(1091,624)),(3,(797,364),(779,352))],
'cbore':    [(4,(697,969),(730,969))],
'plate_front_edge':[(4,(1289,983),(629,982))],
'elbow_motor_len':[(2,(1066,377),(938,377))],
}
out = {}
for k, items in feat.items():
    v = [find(fr, a, b)/S for fr, a, b in items]
    out[k] = dict(mean=round(st.mean(v),1), n=len(v), lo=round(min(v),1), hi=round(max(v),1), vals=[round(x,1) for x in v])
    print(f"{k:16s} mean {out[k]['mean']:6.1f}  n={len(v)}  range {out[k]['lo']}-{out[k]['hi']}  {out[k]['vals']}")

# excluded outliers, for the record
print("\nexcluded: frame1 bracket 'height' ->", round(find(1,(1153,868),(1156,623))/S,1), "(clicked low, bracket top is ~590)")
print("excluded: frame3 forearm 80.3 ->", round(find(3,(829,395),(744,634))/S,1), "(starts at elbow pivot, not slot end)")
print("cross-check turret_h + plate_t =", round(out['turret_h']['mean']+out['plate_t']['mean'],1), "vs frame1 full height", round(find(1,(990,743),(993,868))/S,1))

# shoulder axis height above ground line (puck bottom y=940 in frame 2)
for fr, c in [(1,(1078,638)),(3,(1038,637))]:
    print(f"shoulder axis height frame {fr}: {(940-c[1])/S:.1f} mm")

# link angles (deg from +X, counter-clockwise, image-up positive)
def ang(p_from, p_to):
    dx = p_to[0]-p_from[0]; up = -(p_to[1]-p_from[1])
    return math.degrees(math.atan2(up, dx)) % 360
print("\npose angles (upper, forearm):")
print(" frame1", round(ang((1078,637),(899,378)),1), round(ang((898,377),(586,486)),1))
print(" frame3", round(ang((999,592),(866,440)),1), round(ang((829,395),(744,634)),1))
print(" frame4", round(ang((1042,585),(1130,398)),1), round(ang((1104,317),(930,233)),1))
print("plate front edge at nominal 200 mm ->", round(find(4,(1289,983),(629,982))/200,3), "px/mm (front edge is nearer the camera)")
json.dump(dict(scale=S, feats=out), open('dims.json','w'), indent=1)
// "Digital twin" of one embryo's first days: cells divide one at a time, at the
// division times detected in a real time-lapse video (t2, t3, t4 ... t8, then
// the 8 -> 16 -> 32 rounds before the morula).
//
// The physics (cells pushing each other apart inside the shell) depends on
// history, so it is simulated once, forward, on a fixed time grid. Any time t
// is then an interpolation between two snapshots, which keeps scrubbing the
// timeline backwards and forwards smooth and deterministic.
import * as THREE from 'three'
import { rng } from '../embryo.js'

// Median onset (hours after fertilization) of each phase across the 704 videos
export const TYPICAL = { tPNa: 7.1, tPNf: 24.6, t2: 27.0, t3: 38.0, t4: 39.3, t5: 50.1, t6: 53.0, t7: 55.0, t8: 56.8, 't9+': 71.6, tM: 93.8, tSB: 101.7, tB: 110.3, tEB: 113.3 }

const R0 = 0.66 // zygote radius; the zona's inner surface is at ~0.9
const CONTAINER = 0.88
const STEP = 0.25 // hours between snapshots
const SPLIT_HOURS = 0.9 // how long a division takes on screen

function fill(events) {
  const e = { ...TYPICAL, ...Object.fromEntries(Object.entries(events || {}).filter(([, v]) => Number.isFinite(v))) }
  // keep the order sane if a phase was skipped (e.g. 2 -> 4 directly: no t3)
  if (!Number.isFinite(events?.t3) && Number.isFinite(events?.t4)) e.t3 = e.t4
  const order = ['t2', 't3', 't4', 't5', 't6', 't7', 't8', 't9+', 'tM', 'tSB', 'tB', 'tEB']
  for (let i = 1; i < order.length; i++) e[order[i]] = Math.max(e[order[i]], e[order[i - 1]])
  return e
}

// Division time for every node of the lineage tree (depth 0 = zygote).
function schedule(e, rand) {
  const t9 = e['t9+']
  const lerp = (a, b, u) => a + (b - a) * u
  const times = { '': e.t2, 0: e.t3, 1: e.t4, '00': e.t5, '01': e.t6, 10: e.t7, 11: e.t8 }
  const third = ['000', '001', '010', '011', '100', '101', '110', '111']
  third.forEach((id, i) => { times[id] = lerp(e.t8 + 0.5, t9 + 3, (i + rand() * 0.6) / 8) })
  for (const id of third) {
    for (const b of ['0', '1']) times[id + b] = lerp(t9 + 4, e.tM - 2, rand())
  }
  return times
}

export function buildTwin(events, seed = 9) {
  const rand = rng(seed)
  const e = fill(events)
  const divide = schedule(e, rand)
  const end = e.tM + 1
  const cells = [{ id: '', pos: new THREE.Vector3(), r: R0, born: -1 }]
  const snaps = []
  for (let t = 0; t <= end + 1e-6; t += STEP) {
    // divisions due by time t
    for (let i = cells.length - 1; i >= 0; i--) {
      const c = cells[i]
      if (divide[c.id] !== undefined && divide[c.id] <= t) {
        const axis = new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize()
        const r = c.r * Math.pow(2, -1 / 3) * 1.04
        cells.splice(i, 1,
          { id: c.id + '0', pos: c.pos.clone().addScaledVector(axis, c.r * 0.08), r, born: divide[c.id], axis, sign: 1 },
          { id: c.id + '1', pos: c.pos.clone().addScaledVector(axis, -c.r * 0.08), r, born: divide[c.id], axis, sign: -1 })
      }
    }
    // relax: overlapping cells push apart, everything stays inside the shell
    for (let it = 0; it < 12; it++) {
      for (let a = 0; a < cells.length; a++) {
        for (let b = a + 1; b < cells.length; b++) {
          const d = cells[a].pos.distanceTo(cells[b].pos)
          const min = (cells[a].r + cells[b].r) * 0.84
          if (d < min) {
            const push = new THREE.Vector3().subVectors(cells[a].pos, cells[b].pos)
            if (push.lengthSq() < 1e-9) push.set(rand() - 0.5, rand() - 0.5, rand() - 0.5)
            push.setLength((min - d) * 0.25)
            cells[a].pos.add(push)
            cells[b].pos.sub(push)
          }
        }
        cells[a].pos.multiplyScalar(0.998)
        const lim = CONTAINER - cells[a].r * 0.85
        if (cells[a].pos.length() > lim) cells[a].pos.setLength(lim)
      }
    }
    snaps.push(cells.map((c) => ({ id: c.id, x: c.pos.x, y: c.pos.y, z: c.pos.z, r: c.r, born: c.born, axis: c.axis, sign: c.sign })))
  }

  const idx = (t) => THREE.MathUtils.clamp(t / STEP, 0, snaps.length - 1)

  // Cells at time t. A cell born less than SPLIT_HOURS ago is drawn sliding
  // out of its sister (a dumbbell that pinches apart), not popping in.
  function at(t) {
    const i = Math.floor(idx(t))
    const snap = snaps[i]
    const next = snaps[Math.min(i + 1, snaps.length - 1)]
    const u = idx(t) - i
    const byId = new Map(next.map((c) => [c.id, c]))
    return snap.map((c) => {
      const n = byId.get(c.id) || c
      const pos = new THREE.Vector3(c.x + (n.x - c.x) * u, c.y + (n.y - c.y) * u, c.z + (n.z - c.z) * u)
      let r = c.r
      const age = c.born > 0 ? t - c.born : 99
      if (age < SPLIT_HOURS) {
        const k = THREE.MathUtils.smoothstep(age / SPLIT_HOURS, 0, 1)
        r = c.r * (1.12 - 0.12 * k)
        pos.addScaledVector(c.axis, -c.sign * c.r * 0.55 * (1 - k))
      }
      return { pos, r, id: c.id, age }
    })
  }

  const count = (t) => snaps[Math.floor(idx(t))].length
  return { events: e, at, count, end, splitHours: SPLIT_HOURS }
}

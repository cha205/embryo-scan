// Days 0-4 of development as a deterministic function of time, so the timeline
// can be scrubbed in both directions.
//
// Cells divide without growing, so the lineage tree is precomputed: depth d has
// 2^d cells whose resting positions come from a small physics relaxation inside
// the zona. At time t each cell is interpolated from its parent's position to its own.
import * as THREE from 'three'
import { rng } from '../../src/shared/embryo.js'

export const DIVISIONS = [1.0, 1.6, 2.3, 3.0, 3.4] // days when depth 1..5 begin
const SPLIT = 0.22 // days a division takes on screen
const R0 = 0.7 // zygote radius (zona inner radius is ~0.9)
const CONTAINER = 0.88

const radiusAt = (d) => R0 * Math.pow(2, -d / 3)

function relax(points, r, steps = 400) {
  for (let s = 0; s < steps; s++) {
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const d = points[i].distanceTo(points[j])
        const min = 2 * r * 0.82
        if (d < min) {
          const push = new THREE.Vector3().subVectors(points[i], points[j])
          if (push.lengthSq() < 1e-9) push.set(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)
          push.setLength((min - d) * 0.5)
          points[i].add(push)
          points[j].sub(push)
        }
      }
      points[i].multiplyScalar(0.997) // gentle pull to the centre keeps the cluster compact
      const lim = CONTAINER - r * 0.85
      if (points[i].length() > lim) points[i].setLength(lim)
    }
  }
  return points
}

export function buildLineage(seed = 3) {
  const rand = rng(seed)
  const levels = [[{ pos: new THREE.Vector3(), axis: new THREE.Vector3(0, 1, 0), parent: -1 }]]
  for (let d = 1; d <= DIVISIONS.length; d++) {
    const prev = levels[d - 1]
    const r = radiusAt(d)
    const next = []
    prev.forEach((cell, pi) => {
      const axis = new THREE.Vector3(rand() - 0.5, rand() - 0.5, rand() - 0.5).normalize()
      next.push({ pos: cell.pos.clone().addScaledVector(axis, r * 0.6), axis, parent: pi, sign: 1 })
      next.push({ pos: cell.pos.clone().addScaledVector(axis, -r * 0.6), axis, parent: pi, sign: -1 })
    })
    relax(next.map((c) => c.pos), r)
    levels.push(next)
  }
  return levels
}

const smooth = (x) => x * x * (3 - 2 * x)

// Cells (position + radius) at time t in days, for t < 4.
export function cellsAt(levels, t) {
  let d = 0
  while (d < DIVISIONS.length && t >= DIVISIONS[d]) d++
  const u = d === 0 ? 1 : smooth(Math.min(1, (t - DIVISIONS[d - 1]) / SPLIT))
  const out = []
  for (const cell of levels[d]) {
    if (d === 0) {
      out.push({ pos: cell.pos.clone(), r: radiusAt(0) })
      continue
    }
    const parent = levels[d - 1][cell.parent]
    const start = parent.pos.clone().addScaledVector(cell.axis, cell.sign * radiusAt(d) * 0.05)
    out.push({ pos: start.lerp(cell.pos, u), r: THREE.MathUtils.lerp(radiusAt(d - 1) * 0.86, radiusAt(d), u) })
  }
  return out
}

export const STAGES = [
  { t: 0.0, day: 'Day 0', name: 'Fertilized egg', cells: '1 cell',
    text: 'One cell carrying two sets of DNA, one from each parent. The two round spots inside are those sets, about to merge.' },
  { t: 1.0, day: 'Day 1', name: 'First division', cells: '2 cells',
    text: "The cell splits in two without growing. Every division halves the size of the cells, so the embryo stays the same size." },
  { t: 1.6, day: 'Day 2', name: 'Four cells', cells: '4 cells',
    text: 'Embryologists check that the cells divide on time and come out roughly equal in size.' },
  { t: 2.3, day: 'Day 3', name: 'Eight cells', cells: '8 cells',
    text: 'Eight cells. Around now the embryo switches on its own genes instead of running on the egg\'s supplies.' },
  { t: 3.4, day: 'Day 4', name: 'Morula', cells: '16-32 cells',
    text: 'The cells pull tightly together into a solid ball. Morula is Latin for mulberry.' },
  { t: 4.6, day: 'Day 5', name: 'Blastocyst', cells: '~120 cells',
    text: 'A fluid-filled ball with two kinds of cells: the outer layer becomes the placenta, the inner cluster becomes the baby. This is the stage Embryo Scan grades.' },
]

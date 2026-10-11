// Technical-drawing style labels: an HTML label in a column beside the subject,
// joined to a point on the 3D model by an elbow leader line.
import * as THREE from 'three'

const SVG = 'http://www.w3.org/2000/svg'

export class Callouts {
  constructor(root, camera) {
    this.root = root
    this.camera = camera
    this.svg = document.createElementNS(SVG, 'svg')
    this.svg.classList.add('callout-lines')
    root.appendChild(this.svg)
    this.items = []
    this.visible = true
  }

  // object: the Object3D the point moves with; local: point in its space.
  // side: -1 left column, +1 right column. slot: vertical order in the column.
  add({ object, local, html, side = 1, slot = 0, className = '' }) {
    const el = document.createElement('div')
    el.className = `callout ${className}`
    el.innerHTML = html
    this.root.appendChild(el)
    const line = document.createElementNS(SVG, 'polyline')
    const dot = document.createElementNS(SVG, 'circle')
    dot.setAttribute('r', '3.5')
    this.svg.append(line, dot)
    const item = { object, local: local.clone(), el, line, dot, side, slot }
    this.items.push(item)
    return item
  }

  setVisible(v) {
    this.visible = v
    this.root.classList.toggle('callouts-hidden', !v)
  }

  // Keep label columns clear of these elements (e.g. side panels).
  avoid(leftEls = [], rightEls = []) {
    this.avoidLeft = leftEls
    this.avoidRight = rightEls
  }

  update(center, radius) {
    const w = this.root.clientWidth
    const h = this.root.clientHeight
    const leftLimit = Math.max(8, ...(this.avoidLeft || []).map((e) => e.getBoundingClientRect().right + 12))
    const rightLimit = Math.min(w - 8, ...(this.avoidRight || []).map((e) => e.getBoundingClientRect().left - 12))
    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`)
    const toScreen = (v) => {
      const p = v.clone().project(this.camera)
      return [(p.x + 1) / 2 * w, (1 - p.y) / 2 * h]
    }
    const [cx, cy] = toScreen(center)
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrixWorld, 0)
    const [ex] = toScreen(center.clone().addScaledVector(right, radius))
    const rPx = Math.abs(ex - cx)
    for (const it of this.items) {
      const world = it.local.clone().applyMatrix4(it.object.matrixWorld)
      const [ax, ay] = toScreen(world)
      const lw = it.el.offsetWidth
      let colX = cx + it.side * (rPx + Math.min(150, w * 0.09))
      colX = it.side > 0 ? Math.min(colX, rightLimit - lw - 10) : Math.max(colX, leftLimit + lw + 10)
      const ly = cy - rPx * 0.62 + it.slot * Math.max(64, rPx * 0.55)
      const elbowX = colX - it.side * 28
      it.el.style.transform = `translate(${it.side > 0 ? colX + 10 : colX - 10 - lw}px, ${ly - 14}px)`
      it.line.setAttribute('points', `${ax},${ay} ${elbowX},${ly} ${colX},${ly}`)
      it.dot.setAttribute('cx', ax)
      it.dot.setAttribute('cy', ay)
    }
  }
}

// Develop tab (design B, "Exhibit"): a glass specimen in a culture dish under
// a microscope objective. The 3D twin divides at the times the AI detected in a
// real time-lapse video, which plays alongside.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createBlastocyst, epitheliumMaterial, fluidMaterial } from '../shared/embryo.js'
import { Callouts } from '../shared/callouts.js'
import { buildTwin, TYPICAL } from './twin.js'

// ---------------------------------------------------------------- data
const demo = await fetch('/timelapse/demo.json').then((r) => (r.ok ? r.json() : null)).catch(() => null)
const video = demo?.videos?.[0] ?? null
const events = video?.events_ai ?? TYPICAL
const twin = buildTwin(events, 9)
const E = twin.events
const MAX_H = Math.ceil(Math.max(E.tEB + 10, video?.hours?.at(-1) ?? 0) / 4) * 4

// ---------------------------------------------------------------- renderer
const canvas = document.querySelector('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
renderer.toneMapping = THREE.NeutralToneMapping
renderer.localClippingEnabled = true
const scene = new THREE.Scene()
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture
scene.environmentIntensity = 0.85

const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 200)
camera.position.set(0.7, 1.15, 8.6)
const controls = new OrbitControls(camera, canvas)
controls.target.set(0, -0.2, 0)
controls.enableDamping = true
controls.enablePan = false
controls.minDistance = 4.5
controls.maxDistance = 12
controls.minPolarAngle = 0.55
controls.maxPolarAngle = 1.62
const key = new THREE.DirectionalLight('#ffffff', 2.0)
key.position.set(3, 5, 4)
const rimLight = new THREE.DirectionalLight('#cddcff', 1.4)
rimLight.position.set(-4, 2, -3)
scene.add(key, rimLight)

// ---------------------------------------------------------------- environment
// Studio backdrop: an opaque gradient sphere (transmissive cells refract it)
scene.add(new THREE.Mesh(new THREE.SphereGeometry(60, 64, 32), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false,
  uniforms: { uTop: { value: new THREE.Color('#f4f6f9') }, uBottom: { value: new THREE.Color('#c9d1dc') } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `uniform vec3 uTop, uBottom; varying vec3 vP;
    void main(){ float h = smoothstep(-0.3, 0.5, vP.y); gl_FragColor = vec4(mix(uBottom, uTop, h), 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    }`,
})))

function radialTexture(stops) {
  const c = document.createElement('canvas')
  c.width = c.height = 128
  const g = c.getContext('2d')
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64)
  stops.forEach(([o, col]) => grad.addColorStop(o, col))
  g.fillStyle = grad
  g.fillRect(0, 0, 128, 128)
  const t = new THREE.CanvasTexture(c)
  t.colorSpace = THREE.SRGBColorSpace
  return t
}

// Soft out-of-focus lights far behind the subject, for depth
const bokehTex = radialTexture([[0, 'rgba(255,255,255,0.9)'], [0.55, 'rgba(255,255,255,0.35)'], [1, 'rgba(255,255,255,0)']])
const bokeh = []
for (let i = 0; i < 14; i++) {
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: bokehTex, transparent: true, depthWrite: false, opacity: 0.18 + Math.random() * 0.25, color: i % 3 ? '#ffffff' : '#c9d6ff' }))
  const scale = 1.5 + Math.random() * 4
  s.scale.set(scale, scale, 1)
  s.position.set((Math.random() - 0.5) * 34, -2 + Math.random() * 12, -14 - Math.random() * 14)
  s.userData.speed = 0.05 + Math.random() * 0.1
  bokeh.push(s)
  scene.add(s)
}

const lab = new THREE.Group()
scene.add(lab)
const glass = (opacity, extra = {}) => new THREE.MeshPhysicalMaterial({
  color: '#eef3f8', roughness: 0.06, transparent: true, opacity, depthWrite: false, clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.4, ...extra,
})
// Culture dish and the well the embryo sits in (the bright ring in real videos)
const dish = new THREE.Mesh(new THREE.CylinderGeometry(3.7, 3.75, 0.14, 160), glass(0.32))
dish.position.y = -1.3
const wellWall = new THREE.Mesh(new THREE.CylinderGeometry(1.34, 1.34, 0.32, 128, 1, true), glass(0.22, { side: THREE.DoubleSide }))
wellWall.position.y = -1.08
const wellRim = new THREE.Mesh(new THREE.TorusGeometry(1.34, 0.03, 16, 200), new THREE.MeshPhysicalMaterial({ color: '#ffffff', roughness: 0.15, clearcoat: 1, emissive: '#ffffff', emissiveIntensity: 0.25 }))
wellRim.rotation.x = Math.PI / 2
wellRim.position.y = -0.92
// Culture medium: a faint dome of liquid over the well
const medium = new THREE.Mesh(new THREE.SphereGeometry(2.3, 96, 48, 0, Math.PI * 2, 0, Math.PI / 2), glass(0.035, { color: '#e3ecff', iridescence: 0.4 }))
medium.scale.y = 0.62
medium.position.y = -1.23
// Illumination from below: the bright aperture under the dish and a faint beam
const aperture = new THREE.Mesh(new THREE.CircleGeometry(1.2, 96), new THREE.MeshBasicMaterial({ map: radialTexture([[0, '#ffffff'], [0.7, 'rgba(255,255,255,0.85)'], [1, 'rgba(255,255,255,0)']]), transparent: true, depthWrite: false }))
aperture.rotation.x = -Math.PI / 2
aperture.position.y = -1.38
const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.95, 1.2, 4.2, 96, 1, true), new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: 'varying vec2 vUv; void main(){ gl_FragColor = vec4(1.0, 1.0, 1.0, 0.035 * pow(1.0 - vUv.y, 2.0)); }',
}))
beam.position.y = 0.75
lab.add(dish, wellWall, wellRim, medium, aperture, beam)

// Microscope objective above the specimen
const objective = new THREE.Group()
const metal = (color, roughness = 0.28) => new THREE.MeshStandardMaterial({ color, metalness: 1, roughness })
const barrel = new THREE.Mesh(new THREE.CylinderGeometry(0.64, 0.64, 1.6, 96), metal('#3a404b', 0.35))
barrel.position.y = 0.8
const band = new THREE.Mesh(new THREE.CylinderGeometry(0.648, 0.648, 0.09, 96), new THREE.MeshStandardMaterial({ color: '#2f5bff', metalness: 0.3, roughness: 0.3 }))
band.position.y = 0.32
const nose = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.6, 0.42, 96), metal('#d5d9df', 0.14))
nose.position.y = -0.21
const lens = new THREE.Mesh(new THREE.CircleGeometry(0.3, 64), new THREE.MeshPhysicalMaterial({ color: '#9fb5ff', metalness: 0, roughness: 0.02, clearcoat: 1, iridescence: 1, envMapIntensity: 2 }))
lens.rotation.x = Math.PI / 2
lens.position.y = -0.425
objective.add(barrel, band, nose, lens)
objective.position.y = 1.95
lab.add(objective)

// Specks suspended in the medium
{
  const n = 260
  const pos = new Float32Array(n * 3)
  const seed = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2
    const r = 1.2 + Math.random() * 2.2
    pos.set([Math.cos(a) * r, -1.1 + Math.random() * 2.6, Math.sin(a) * r], i * 3)
    seed[i] = Math.random()
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  g.setAttribute('seed', new THREE.BufferAttribute(seed, 1))
  const specks = new THREE.Points(g, fluidMaterial({ uTime: { value: 0 } }, '#7d8fb3', 1.6, 0.45))
  specks.material.blending = THREE.NormalBlending
  specks.name = 'specks'
  lab.add(specks)
}

// ---------------------------------------------------------------- the embryo
const cut = new THREE.Plane(new THREE.Vector3(0, 0, -1), 10)
const specimen = new THREE.Group()
specimen.position.y = -0.05
scene.add(specimen)

const zonaMat = new THREE.MeshPhysicalMaterial({
  color: '#ffffff', roughness: 0.04, transparent: true, opacity: 0.08, depthWrite: false,
  iridescence: 0.9, iridescenceIOR: 1.3, iridescenceThicknessRange: [200, 600],
  clearcoat: 1, clearcoatRoughness: 0.05, envMapIntensity: 1.6, clippingPlanes: [cut],
})
zonaMat.onBeforeCompile = (shader) => {
  shader.fragmentShader = shader.fragmentShader.replace('#include <opaque_fragment>', `
    float edge = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), 2.0);
    diffuseColor.a = mix(diffuseColor.a, 0.85, edge);
    #include <opaque_fragment>`)
}
const zona = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), zonaMat)
zona.renderOrder = 2
specimen.add(zona)

// Cells: frosted, see-through (transmission) so the nuclei inside show
const cellMat = new THREE.MeshPhysicalMaterial({
  color: '#f7f9fc', transmission: 0.92, thickness: 0.35, roughness: 0.14, ior: 1.33,
  attenuationColor: '#d3ddef', attenuationDistance: 2.5, clearcoat: 0.6, clearcoatRoughness: 0.15,
  sheen: 0.4, sheenColor: '#dbe6ff', iridescence: 0.12, clippingPlanes: [cut],
})
const cells = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 48, 32), cellMat, 32)
const nucMat = new THREE.MeshStandardMaterial({ color: '#5b70ad', roughness: 0.5 })
const nuclei = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 24, 16), nucMat, 32)
specimen.add(cells, nuclei)
const pronuclei = new THREE.Group()
for (const x of [-0.17, 0.17]) {
  const p = new THREE.Mesh(new THREE.SphereGeometry(0.15, 32, 24), new THREE.MeshStandardMaterial({ color: '#6f86c4', roughness: 0.5 }))
  p.position.set(x, 0.04, 0.05)
  pronuclei.add(p)
}
specimen.add(pronuclei)

const fluid = fluidMaterial({ uTime: { value: 0 } }, '#4c6fff', 2.2, 0.5)
fluid.blending = THREE.NormalBlending
const blast = createBlastocyst({
  expansion: 4, icm: 'A', te: 'A', seed: 5, teSize: 0.92, teFlat: 0.03,
  materials: {
    zona: new THREE.MeshBasicMaterial({ visible: false }),
    te: new THREE.MeshPhysicalMaterial({
      color: '#f5f8fc', transmission: 0.8, thickness: 0.25, roughness: 0.3, ior: 1.36, transparent: true,
      clearcoat: 0.6, sheen: 0.5, sheenColor: '#dbe6ff', clippingPlanes: [cut],
    }),
    icm: new THREE.MeshPhysicalMaterial({ color: '#5277ff', roughness: 0.3, clearcoat: 0.8, sheen: 0.6, sheenColor: '#9fb6ff', transparent: true }),
    fluid,
  },
})
blast.userData.parts.zona.visible = false
specimen.add(blast)
const { anchors, parts, teRadius } = blast.userData
// one continuous trophectoderm sheet (cell borders + nuclei drawn by shader)
parts.te.visible = false
const teSheet = new THREE.Mesh(new THREE.SphereGeometry(teRadius, 160, 120), epitheliumMaterial({
  freq: 3.7, // ~170 cells over the sphere, like a grade-A trophectoderm
  color: '#f4f7fc', transmission: 0.7, thickness: 0.12, roughness: 0.25, ior: 1.36, transparent: true,
  clearcoat: 0.7, clearcoatRoughness: 0.15, sheen: 0.5, sheenColor: '#dbe6ff', side: THREE.DoubleSide, clippingPlanes: [cut],
}))
blast.add(teSheet)
const teNuclei = { material: { opacity: 0 } } // nuclei now live in the sheet's shader

const callouts = new Callouts(document.querySelector('#callouts'), camera)
callouts.add({ object: blast, local: anchors.icm, side: 1, slot: 0, className: 'accent', html: 'Inner cell mass<span>becomes the baby</span>' })
callouts.add({ object: blast, local: anchors.te, side: 1, slot: 1, html: 'Trophectoderm<span>becomes the placenta</span>' })
callouts.add({ object: blast, local: anchors.cavity, side: 1, slot: 2, html: 'Blastocoel<span>fluid-filled cavity</span>' })
callouts.add({ object: zona, local: anchors.zona, side: 1, slot: 3, html: 'Zona pellucida<span>protective shell</span>' })

// ---------------------------------------------------------------- stages
const STAGES = [
  { from: 0, name: 'Fertilized egg', text: 'One cell. The two round spots are the sets of DNA from each parent, before they merge.' },
  { from: E.t2, name: 'Two cells', text: 'The first division. Cells split without growing, so each division halves their size.' },
  { from: E.t3, name: 'Three cells', text: 'Divisions are not simultaneous: one cell splits before its sister. A long gap here can be a warning sign.' },
  { from: E.t4, name: 'Four cells', text: 'The second round is done. How close together t3 and t4 happen (s2) is a known quality marker.' },
  { from: E.t5, name: 'Five to eight cells', text: 'The four cells divide one by one. When the fifth cell appears (t5) is one of the strongest timing markers.' },
  { from: E.t8, name: 'Eight cells', text: 'Day 3. The embryo starts using its own genes instead of the egg\'s supplies.' },
  { from: E['t9+'], name: 'Nine or more cells', text: 'Divisions speed up and fall out of sync; individual cells get hard to count.' },
  { from: E.tM, name: 'Morula', text: 'The cells pull tightly together into a solid ball. Morula is Latin for mulberry.' },
  { from: E.tSB, name: 'Blastocyst forming', text: 'Fluid starts collecting inside. The cells split into an outer layer and an inner cluster.' },
  { from: E.tB, name: 'Blastocyst', text: 'Day 5. The outer layer becomes the placenta, the inner cluster becomes the baby. This is the stage Embryo Scan grades.' },
]
const MARKS = [['t2', '2'], ['t3', '3'], ['t4', '4'], ['t5', '5'], ['t8', '8'], ['t9+', '9+'], ['tM', 'M'], ['tSB', 'SB'], ['tB', 'B']]
const stageAt = (h) => { let s = 0; STAGES.forEach((st, i) => { if (h >= st.from) s = i }); return s }

// ---------------------------------------------------------------- UI
const $ = (s) => document.querySelector(s)
const state = { h: 0, target: 0, playing: false, dragging: false, inside: false, insideAmt: 0, stage: -1, frame: -1 }
const track = $('#track')

for (let d = 0; d <= MAX_H; d += 24) {
  const s = document.createElement('span')
  s.style.left = `${(d / MAX_H) * 100}%`
  s.textContent = `Day ${d / 24}`
  $('#days').append(s)
}
const trackMarks = MARKS.map(([k, label]) => {
  const s = document.createElement('span')
  s.style.left = `${(E[k] / MAX_H) * 100}%`
  s.textContent = label
  $('#track-events').append(s)
  return [k, s]
})
const eventItems = MARKS.slice(0, 8).map(([k, label]) => {
  const li = document.createElement('li')
  li.innerHTML = `${label === 'M' ? 'morula' : label === 'SB' ? 'blast.' : label + ' cells'}<b>${E[k].toFixed(1)} h</b>`
  $('#rv-events').append(li)
  return [k, li]
})

// cell-count step chart (log2 scale: 1, 2, 4, 8, 16, 32)
const chartH = Math.min(MAX_H, E.tM + 4)
$('#rv-axis-end').textContent = `${Math.round(chartH)} h`
const svg = $('#rv-chart')
const NS = 'http://www.w3.org/2000/svg'
const cy = (n) => 86 - (Math.log2(Math.max(1, n)) / 5) * 80
for (const n of [1, 2, 4, 8, 16, 32]) {
  const l = document.createElementNS(NS, 'line')
  l.setAttribute('class', 'grid')
  l.setAttribute('x1', 0); l.setAttribute('x2', 300); l.setAttribute('y1', cy(n)); l.setAttribute('y2', cy(n))
  svg.append(l)
}
const area = document.createElementNS(NS, 'path')
area.setAttribute('class', 'area')
const step = document.createElementNS(NS, 'path')
step.setAttribute('class', 'step')
const head = document.createElementNS(NS, 'line')
head.setAttribute('class', 'head')
head.setAttribute('y1', 0); head.setAttribute('y2', 92)
svg.append(area, step, head)
const counts = []
for (let h = 0; h <= chartH; h += 0.5) counts.push([h, twin.count(h)])
function drawChart(h) {
  const pts = counts.filter(([t]) => t <= h)
  if (!pts.length) return
  let d = `M0 ${cy(pts[0][1])}`
  for (let i = 1; i < pts.length; i++) {
    const x = (pts[i][0] / chartH) * 300
    d += ` H${x.toFixed(1)} V${cy(pts[i][1]).toFixed(1)}`
  }
  const xEnd = (Math.min(h, chartH) / chartH) * 300
  d += ` H${xEnd.toFixed(1)}`
  step.setAttribute('d', d)
  area.setAttribute('d', `${d} V86 H0 Z`)
  head.setAttribute('x1', xEnd); head.setAttribute('x2', xEnd)
}

// real video: one WebM per embryo, frame i stored at i / fps seconds; the
// player never plays, it seeks to the frame recorded nearest the timeline hour
const player = $('#rv-video')
if (video) {
  $('#rv-empty').style.display = 'none'
  $('#rv-id').textContent = video.id
  player.src = video.video_url
} else {
  player.style.display = 'none'
}
function showFrame(h) {
  if (!video) return
  const hours = video.hours
  let lo = 0
  let hi = hours.length - 1
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (hours[mid] <= h) lo = mid; else hi = mid - 1 }
  if (lo !== state.frame && !player.seeking) {
    state.frame = lo
    player.currentTime = (lo + 0.5) / video.fps
  }
}

function setPlaying(p) {
  state.playing = p
  $('#btn-play').innerHTML = `<i class="ti ti-player-${p ? 'pause' : 'play'}"></i>`
}
$('#btn-play').addEventListener('click', () => {
  if (!state.playing && state.h >= MAX_H - 0.1) state.h = state.target = 0
  setPlaying(!state.playing)
})
const hFromPointer = (e) => {
  const r = track.getBoundingClientRect()
  return THREE.MathUtils.clamp((e.clientX - r.left) / r.width, 0, 1) * MAX_H
}
track.addEventListener('pointerdown', (e) => {
  state.dragging = true
  setPlaying(false)
  track.setPointerCapture(e.pointerId)
  state.h = state.target = hFromPointer(e)
})
track.addEventListener('pointermove', (e) => { if (state.dragging) state.h = state.target = hFromPointer(e) })
track.addEventListener('pointerup', () => { state.dragging = false })
$('#btn-inside').addEventListener('click', () => {
  state.inside = !state.inside
  $('#btn-inside span').textContent = state.inside ? 'Close' : 'Look inside'
  $('#btn-inside i').className = `ti ti-${state.inside ? 'eye-off' : 'eye'}`
  document.querySelector('.real').style.opacity = state.inside ? 0 : 1
})

const story = document.querySelector('.story')
function showStage(i) {
  const s = STAGES[i]
  story.classList.add('swap')
  setTimeout(() => {
    $('#st-name').textContent = s.name
    $('#st-text').textContent = s.text
    $('#st-grades').classList.toggle('show', i >= STAGES.length - 1)
    story.classList.remove('swap')
  }, 200)
}

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  camera.setViewOffset(w, h, 0, h * 0.04, w, h)
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

// ---------------------------------------------------------------- loop
const smooth = (x) => { x = THREE.MathUtils.clamp(x, 0, 1); return x * x * (3 - 2 * x) }
const m = new THREE.Matrix4()
const q = new THREE.Quaternion()
const v3 = new THREE.Vector3()
const clock = new THREE.Clock()
let last = 0
setTimeout(() => setPlaying(true), 900)
const SPEED = MAX_H / 26 // the whole development in ~26 seconds

renderer.setAnimationLoop(() => {
  const now = clock.getElapsedTime()
  const dt = Math.min(1, now - last)
  last = now
  fluid.uniforms.uTime.value = now
  lab.getObjectByName('specks').material.uniforms.uTime.value = now

  if (state.playing) {
    state.h = state.target = Math.min(MAX_H, state.h + dt * SPEED)
    if (state.h >= MAX_H) setPlaying(false)
  } else if (!state.dragging) {
    state.h += (state.target - state.h) * Math.min(1, dt * 6)
  }
  const h = state.h
  const pct = (h / MAX_H) * 100
  $('#handle').style.left = `${pct}%`
  $('#fill').style.width = `${pct}%`
  $('#clock-hours').textContent = `${h.toFixed(1)} h`
  $('#clock-day').textContent = `Day ${Math.floor(h / 24)}`
  $('#st-day').textContent = `Day ${Math.floor(h / 24)} · ${h.toFixed(0)} h`
  $('#rv-time').textContent = `${h.toFixed(1)} h`
  const si = stageAt(h)
  if (si !== state.stage) { state.stage = si; showStage(si) }
  $('#st-cells').textContent = h >= E.tSB ? '~120' : h >= E['t9+'] ? `${twin.count(h)}+` : twin.count(h)
  trackMarks.forEach(([k, s]) => s.classList.toggle('done', h >= E[k]))
  let nowKey = null
  eventItems.forEach(([k, li]) => { if (h >= E[k]) nowKey = k })
  eventItems.forEach(([k, li]) => { li.classList.toggle('done', h >= E[k]); li.classList.toggle('now', k === nowKey && h - E[k] < 6) })
  const next = MARKS.find(([k]) => E[k] > h)
  $('#st-next').textContent = next ? `${E[next[0]].toFixed(1)} h` : 'done'
  $('#st-next-label').textContent = next ? `next: ${next[1] === 'M' ? 'morula' : next[1] === 'SB' ? 'blastocyst' : next[1] + ' cells'}` : 'development'
  drawChart(h)
  showFrame(h)

  // cleavage cells until the morula, then the blastocyst takes over
  const form = smooth((h - E.tSB) / Math.max(2, E.tB - E.tSB))
  const compact = 1 + 0.18 * smooth((h - (E.tM - 4)) / 4)
  const list = twin.at(Math.min(h, twin.end))
  cells.count = nuclei.count = list.length
  list.forEach((c, i) => {
    const r = c.r * compact * (1 - form)
    v3.copy(c.pos).multiplyScalar(1 - 0.3 * form)
    m.compose(v3, q, new THREE.Vector3(r, r, r))
    cells.setMatrixAt(i, m)
    // the nucleus dissolves while a cell divides (mitosis) and reforms after
    const nk = (c.age < twin.splitHours ? smooth((c.age / twin.splitHours - 0.55) / 0.45) : 1) * (h < E.tPNf ? 0 : 1)
    const nr = r * 0.34 * nk
    m.compose(v3, q, new THREE.Vector3(nr, nr, nr))
    nuclei.setMatrixAt(i, m)
  })
  cells.instanceMatrix.needsUpdate = true
  nuclei.instanceMatrix.needsUpdate = true
  cells.visible = nuclei.visible = form < 0.999
  pronuclei.visible = h < E.t2
  const pn = smooth((h - (E.tPNa - 2)) / 3) * (1 - smooth((h - E.tPNf + 1) / 2))
  pronuclei.children.forEach((p, i) => {
    p.scale.setScalar(Math.max(0.001, pn))
    p.position.x = (i ? 1 : -1) * (0.17 - 0.12 * smooth((h - 12) / (E.tPNf - 12)))
  })

  blast.visible = form > 0.001
  const expand = smooth((h - E.tB) / Math.max(3, E.tEB - E.tB + 4))
  blast.scale.setScalar(THREE.MathUtils.lerp(0.62, 0.95, form) + 0.04 * expand)
  teSheet.material.opacity = form
  teNuclei.material.opacity = form
  parts.icm.material.opacity = smooth((h - E.tSB - 2) / Math.max(2, E.tB - E.tSB))
  parts.fluid.material.uniforms.uOpacity.value = 0.5 * form
  zona.scale.setScalar(1 + 0.05 * expand)

  // look inside: peel away the near side of the shell (blastocyst only)
  const insideTarget = state.inside && si === STAGES.length - 1 ? 1 : 0
  state.insideAmt += (insideTarget - state.insideAmt) * Math.min(1, dt * 4)
  cut.normal.copy(v3.subVectors(camera.position, controls.target).normalize()).negate()
  cut.constant = THREE.MathUtils.lerp(1.3, 0.16, smooth(state.insideAmt))
  callouts.setVisible(state.insideAmt > 0.6)

  bokeh.forEach((b) => { b.position.x += b.userData.speed * dt * 0.3; if (b.position.x > 18) b.position.x = -18 })
  specimen.rotation.y += dt * 0.12
  specimen.position.y = -0.05 + 0.04 * Math.sin(now * 0.8)
  objective.position.y = 1.95 + 0.02 * Math.sin(now * 0.5)
  controls.update()
  renderer.render(scene, camera)
  if (state.insideAmt > 0.05) callouts.update(specimen.position, 1.0)
})

// Direction B: "Exhibit" — a bright science-museum piece. A glass embryo on a
// plinth develops from one cell to a blastocyst as you scrub the timeline.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js'
import { createBlastocyst, fluidMaterial } from './embryo.js'
import { buildLineage, cellsAt, STAGES } from './development.js'
import { Callouts } from './callouts.js'

const canvas = document.querySelector('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5))
renderer.toneMapping = THREE.NeutralToneMapping
renderer.localClippingEnabled = true

const scene = new THREE.Scene()
const pmrem = new THREE.PMREMGenerator(renderer)
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
scene.environmentIntensity = 0.9

// Studio backdrop (an opaque sphere, so the glass has something to refract)
scene.add(new THREE.Mesh(new THREE.SphereGeometry(40, 64, 32), new THREE.ShaderMaterial({
  side: THREE.BackSide,
  depthWrite: false,
  uniforms: { uTop: { value: new THREE.Color('#f5f7fa') }, uBottom: { value: new THREE.Color('#cfd6df') } },
  vertexShader: 'varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: `uniform vec3 uTop, uBottom; varying vec3 vP;
    void main(){ float h = smoothstep(-0.25, 0.45, vP.y); gl_FragColor = vec4(mix(uBottom, uTop, h), 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
    }`,
})))

const camera = new THREE.PerspectiveCamera(28, 1, 0.1, 100)
camera.position.set(0.4, 0.7, 8.2)
const controls = new OrbitControls(camera, canvas)
controls.target.set(0, -0.15, 0)
controls.enableDamping = true
controls.enablePan = false
controls.minDistance = 4.5
controls.maxDistance = 10
controls.minPolarAngle = 0.6
controls.maxPolarAngle = 1.75

const key = new THREE.DirectionalLight('#ffffff', 2.2)
key.position.set(3, 4, 5)
const rim = new THREE.DirectionalLight('#cddcff', 1.6)
rim.position.set(-4, 2, -3)
scene.add(key, rim)

// Plinth with a thin accent ring and a soft contact shadow
const plinth = new THREE.Group()
plinth.position.y = -2.05
plinth.add(new THREE.Mesh(new THREE.CylinderGeometry(0.95, 1.0, 0.7, 128), new THREE.MeshPhysicalMaterial({ color: '#f8f9fb', roughness: 0.5, clearcoat: 0.4 })))
const ring = new THREE.Mesh(new THREE.TorusGeometry(0.953, 0.005, 8, 256), new THREE.MeshBasicMaterial({ color: '#2f5bff', transparent: true, opacity: 0.7 }))
ring.rotation.x = Math.PI / 2
ring.position.y = 0.35
plinth.add(ring)
{
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const g = c.getContext('2d')
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 128)
  grad.addColorStop(0, 'rgba(20,30,50,0.55)')
  grad.addColorStop(0.55, 'rgba(20,30,50,0.18)')
  grad.addColorStop(1, 'rgba(20,30,50,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, 256, 256)
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1.8, 1.8), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }))
  shadow.rotation.x = -Math.PI / 2
  shadow.position.y = 0.352
  plinth.add(shadow)
}
scene.add(plinth)

// --- the specimen -------------------------------------------------------------
const cut = new THREE.Plane(new THREE.Vector3(0, 0, -1), 10)
const specimen = new THREE.Group()
scene.add(specimen)

// Glass shell: mostly clear face-on, whiter and more reflective at the edges
// (Fresnel). Plain transparency instead of three.js "transmission", which
// would hide the cells inside and costs an extra render pass.
const zonaMat = new THREE.MeshPhysicalMaterial({
  color: '#ffffff', roughness: 0.04, metalness: 0, transparent: true, opacity: 0.08, depthWrite: false,
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

const pearl = (color, extra = {}) => new THREE.MeshPhysicalMaterial({
  color, roughness: 0.3, clearcoat: 0.8, clearcoatRoughness: 0.18, sheen: 0.7, sheenColor: '#c5d6ff',
  sheenRoughness: 0.45, iridescence: 0.25, iridescenceIOR: 1.3, ...extra,
})

const levels = buildLineage(4)
const cellMat = pearl('#eef2f7', { transparent: true })
const cells = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 40, 28), cellMat, 32)
specimen.add(cells)
const pronuclei = new THREE.Group()
for (const x of [-0.14, 0.14]) {
  // drawn after the half-transparent zygote and without depth testing, so
  // they show through its surface
  const p = new THREE.Mesh(new THREE.SphereGeometry(0.12, 48, 32), pearl('#6d8bff', { transparent: true, opacity: 0.85, depthTest: false }))
  p.position.set(x, 0.04, 0.12)
  p.renderOrder = 3
  pronuclei.add(p)
}
specimen.add(pronuclei)

const fluid = fluidMaterial({ uTime: { value: 0 } }, '#4c6fff', 2.4, 0.5)
fluid.blending = THREE.NormalBlending
const uniforms = fluid.uniforms
const blast = createBlastocyst({
  expansion: 4, icm: 'A', te: 'A', seed: 5, teSize: 0.92, teFlat: 0.03,
  materials: {
    zona: new THREE.MeshBasicMaterial({ visible: false }),
    te: pearl('#f1f4f9', { transparent: true, clippingPlanes: [cut] }),
    icm: pearl('#5277ff', { transparent: true, sheenColor: '#9fb6ff' }),
    fluid,
  },
})
blast.userData.parts.zona.visible = false
specimen.add(blast)
const { anchors, parts } = blast.userData

// --- labels for "look inside" ------------------------------------------------------
const callouts = new Callouts(document.querySelector('#callouts'), camera)
callouts.add({ object: blast, local: anchors.icm, side: 1, slot: 0, className: 'accent', html: 'Inner cell mass<span>becomes the baby</span>' })
callouts.add({ object: blast, local: anchors.te, side: 1, slot: 1, html: 'Trophectoderm<span>becomes the placenta</span>' })
callouts.add({ object: blast, local: anchors.cavity, side: 1, slot: 2, html: 'Blastocoel<span>fluid-filled cavity</span>' })
callouts.add({ object: zona, local: anchors.zona, side: 1, slot: 3, html: 'Zona pellucida<span>protective shell</span>' })

// --- timeline UI ------------------------------------------------------------------
const MAX_T = 5
const state = { t: 0, target: 0, playing: false, dragging: false, inside: false, insideAmt: 0, stage: -1 }
const track = document.querySelector('#track')
const handle = document.querySelector('#handle')
const fill = document.querySelector('#fill')
const marks = document.querySelector('#marks')
const playBtn = document.querySelector('#btn-play')
const insideBtn = document.querySelector('#btn-inside')

STAGES.forEach((s, i) => {
  const tick = document.createElement('i')
  tick.style.left = `${(s.t / MAX_T) * 100}%`
  const b = document.createElement('button')
  b.style.left = `${(s.t / MAX_T) * 100}%`
  b.innerHTML = `${s.name}<small>${s.day}</small>`
  b.addEventListener('click', (e) => { e.stopPropagation(); setPlaying(false); state.target = s.t + (i === STAGES.length - 1 ? 0.4 : 0.05) })
  marks.append(tick, b)
})

function setPlaying(p) {
  state.playing = p
  playBtn.innerHTML = `<i class="ti ti-player-${p ? 'pause' : 'play'}"></i>`
}
playBtn.addEventListener('click', () => {
  if (!state.playing && state.t >= MAX_T - 0.01) { state.t = state.target = 0 }
  setPlaying(!state.playing)
})
const tFromPointer = (e) => {
  const r = track.getBoundingClientRect()
  return THREE.MathUtils.clamp((e.clientX - r.left) / r.width, 0, 1) * MAX_T
}
track.addEventListener('pointerdown', (e) => {
  state.dragging = true
  setPlaying(false)
  track.setPointerCapture(e.pointerId)
  state.t = state.target = tFromPointer(e)
})
track.addEventListener('pointermove', (e) => { if (state.dragging) state.t = state.target = tFromPointer(e) })
track.addEventListener('pointerup', () => { state.dragging = false })
insideBtn.addEventListener('click', () => {
  state.inside = !state.inside
  insideBtn.querySelector('span').textContent = state.inside ? 'Close' : 'Look inside'
  insideBtn.querySelector('i').className = `ti ti-${state.inside ? 'eye-off' : 'eye'}`
})

const story = document.querySelector('.story')
function showStage(i) {
  const s = STAGES[i]
  story.classList.add('swap')
  setTimeout(() => {
    document.querySelector('#st-day').textContent = s.day
    document.querySelector('#st-name').textContent = s.name
    document.querySelector('#st-text').textContent = s.text
    document.querySelector('#st-cells').textContent = s.cells
    document.querySelector('#st-grades').classList.toggle('show', i === STAGES.length - 1)
    story.classList.remove('swap')
  }, 220)
  marks.querySelectorAll('button').forEach((b, j) => b.classList.toggle('on', j === i))
}

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  // push the specimen right of centre, clear of the story panel
  camera.setViewOffset(w, h, w > 900 ? -w * 0.11 : 0, h * 0.06, w, h)
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

// --- animation -------------------------------------------------------------------
const smooth = (x) => { x = THREE.MathUtils.clamp(x, 0, 1); return x * x * (3 - 2 * x) }
const m = new THREE.Matrix4()
const q = new THREE.Quaternion()
const clock = new THREE.Clock()
let last = 0
setTimeout(() => setPlaying(true), 900)

renderer.setAnimationLoop(() => {
  const now = clock.getElapsedTime()
  const dt = Math.min(1, now - last)
  last = now
  uniforms.uTime.value = now

  if (state.playing) {
    state.t = state.target = Math.min(MAX_T, state.t + dt * 0.42)
    if (state.t >= MAX_T) setPlaying(false)
  } else if (!state.dragging) {
    state.t += (state.target - state.t) * Math.min(1, dt * 6)
  }
  const t = state.t
  const pct = (t / MAX_T) * 100
  handle.style.left = `${pct}%`
  fill.style.width = `${pct}%`
  let si = 0
  STAGES.forEach((s, i) => { if (t >= s.t - 0.02) si = i })
  if (si !== state.stage) { state.stage = si; showStage(si) }

  // cleavage cells (day 0-4), fading into the blastocyst from day 3.9
  const form = smooth((t - 3.9) / 0.7)
  const compact = 1 + 0.16 * smooth((t - 3.3) / 0.5)
  const list = cellsAt(levels, Math.min(t, 3.99))
  cells.count = list.length
  list.forEach((c, i) => {
    const r = c.r * compact * (1 - form)
    m.compose(c.pos.clone().multiplyScalar(1 - 0.25 * form), q, new THREE.Vector3(r, r, r))
    cells.setMatrixAt(i, m)
  })
  cells.instanceMatrix.needsUpdate = true
  cells.visible = form < 0.999
  cellMat.opacity = t < 1 ? 0.55 + 0.45 * smooth((t - 0.75) / 0.25) : 1
  pronuclei.visible = t < 1
  pronuclei.children.forEach((p, i) => p.position.x = (i ? 1 : -1) * (0.14 - 0.1 * smooth((t - 0.5) / 0.45)))

  blast.visible = form > 0.001
  blast.scale.setScalar(THREE.MathUtils.lerp(0.62, 0.97, form) + 0.03 * smooth((t - 4.6) / 0.4))
  parts.te.material.opacity = form
  parts.icm.material.opacity = smooth((t - 4.2) / 0.5)
  parts.fluid.material.uniforms.uOpacity.value = 0.5 * form
  zona.scale.setScalar(1 + 0.04 * smooth((t - 4.5) / 0.5))

  // look inside: peel away the near side of the shell
  state.insideAmt += ((state.inside && si === STAGES.length - 1 ? 1 : 0) - state.insideAmt) * Math.min(1, dt * 4)
  const camDir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize()
  cut.normal.copy(camDir).negate()
  cut.constant = THREE.MathUtils.lerp(1.2, 0.18, smooth(state.insideAmt))
  callouts.setVisible(state.insideAmt > 0.6)

  specimen.rotation.y += dt * 0.18
  specimen.position.y = 0.05 * Math.sin(now * 0.8)
  controls.update()
  renderer.render(scene, camera)
  if (state.insideAmt > 0.05) callouts.update(specimen.position, 1.0)
})

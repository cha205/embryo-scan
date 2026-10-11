// Direction A: "Specimen" — a holographic lab instrument that scans a 3D
// blastocyst and reports the grade and live-birth estimate.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js'
import { createBlastocyst, fluidMaterial, patchHolo } from './embryo.js'
import { Callouts } from './callouts.js'

const CYAN = new THREE.Color('#5fe3ff')
const BG = '#050a12'

const canvas = document.querySelector('#scene')
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.toneMapping = THREE.ACESFilmicToneMapping
renderer.toneMappingExposure = 1.05
renderer.localClippingEnabled = true

const scene = new THREE.Scene()
scene.background = new THREE.Color(BG)
scene.fog = new THREE.FogExp2(BG, 0.075)
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100)
const target = new THREE.Vector3(0, -0.12, 0)
const startPos = new THREE.Vector3(0.6, 1.4, 9.5)
const homePos = new THREE.Vector3(0.0, 0.55, 5.4)
camera.position.copy(startPos)
const controls = new OrbitControls(camera, canvas)
controls.target.copy(target)
controls.enableDamping = true
controls.enablePan = false
controls.minDistance = 3.4
controls.maxDistance = 9
controls.enabled = false

const uniforms = { uTime: { value: 0 }, uScanY: { value: 99 } }
const cut = new THREE.Plane(new THREE.Vector3(0, 0, -1), 10)

scene.add(new THREE.AmbientLight('#6fa8ff', 0.35))
const key = new THREE.PointLight('#9feaff', 26, 12, 1.6)
key.position.set(2.6, 2.4, 3.2)
const back = new THREE.PointLight('#2f5dff', 20, 12, 1.6)
back.position.set(-3, -0.5, -2.5)
const warm = new THREE.PointLight('#ffb547', 3.5, 3, 1.5)
scene.add(key, back, warm)

// --- materials -------------------------------------------------------------
function shellMaterial(strength, side) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    side,
    blending: THREE.AdditiveBlending,
    clipping: true,
    clippingPlanes: [cut],
    uniforms: { ...uniforms, uColor: { value: CYAN }, uStrength: { value: strength } },
    vertexShader: /* glsl */`
      #include <clipping_planes_pars_vertex>
      varying vec3 vN; varying vec3 vV; varying vec3 vW;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal);
        vV = -mvPosition.xyz;
        vW = (modelMatrix * vec4(position, 1.0)).xyz;
        gl_Position = projectionMatrix * mvPosition;
        #include <clipping_planes_vertex>
      }`,
    fragmentShader: /* glsl */`
      #include <clipping_planes_pars_fragment>
      uniform vec3 uColor; uniform float uStrength, uScanY, uTime;
      varying vec3 vN; varying vec3 vV; varying vec3 vW;
      void main() {
        #include <clipping_planes_fragment>
        float f = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.4);
        float band = exp(-pow((vW.y - uScanY) / 0.025, 2.0));
        float lat = smoothstep(0.965, 1.0, abs(sin(vW.y * 34.0))) * 0.10;
        float shimmer = 0.85 + 0.15 * sin(uTime * 0.8 + vW.y * 3.0);
        vec3 c = uColor * (f * uStrength * shimmer + band * 1.2 + lat * f);
        gl_FragColor = vec4(c, 1.0);
      }`,
  })
}

const materials = {
  zona: shellMaterial(1.25, THREE.FrontSide),
  zonaInner: shellMaterial(0.45, THREE.BackSide),
  te: patchHolo(new THREE.MeshStandardMaterial({
    color: '#0f3346', roughness: 0.3, metalness: 0.15, transparent: true, opacity: 0.5, depthWrite: false,
    emissive: '#05202e', clippingPlanes: [cut],
  }), uniforms, { rim: '#5fe3ff', strength: 0.9, power: 2.0 }),
  icm: patchHolo(new THREE.MeshStandardMaterial({
    color: '#5a3209', roughness: 0.35, metalness: 0.1, emissive: '#ff9b2f', emissiveIntensity: 0.32,
  }), uniforms, { rim: '#ffd08a', strength: 1.3, scan: '#ffe7c2' }),
  fluid: fluidMaterial(uniforms, '#7fe6ff', 2.2, 0.42),
}

const embryo = createBlastocyst({ expansion: 4, icm: 'A', te: 'A', seed: 11, materials })
scene.add(embryo)
const { anchors, parts, teRadius } = embryo.userData
warm.position.copy(anchors.icm).multiplyScalar(0.6)

// Glowing nuclei, one per trophectoderm cell
{
  const cells = parts.te.userData.cells
  const nuclei = new THREE.InstancedMesh(new THREE.SphereGeometry(0.011, 8, 6),
    new THREE.MeshBasicMaterial({ color: '#aef4ff', toneMapped: false, clippingPlanes: [cut], transparent: true, opacity: 0.85 }), cells.length)
  const m = new THREE.Matrix4()
  cells.forEach((c, i) => nuclei.setMatrixAt(i, m.makeTranslation(c.normal.clone().multiplyScalar(teRadius - 0.02))))
  embryo.add(nuclei)
}

// --- scanner ring and platform ---------------------------------------------
const scanGroup = new THREE.Group()
const scanRing = new THREE.Mesh(new THREE.TorusGeometry(1, 0.006, 8, 220), new THREE.MeshBasicMaterial({ color: '#c9f8ff', toneMapped: false }))
scanRing.rotation.x = Math.PI / 2
const sheet = new THREE.Mesh(new THREE.CircleGeometry(1, 96), new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  uniforms: { uColor: { value: CYAN } },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
  fragmentShader: 'uniform vec3 uColor; varying vec2 vUv; void main(){ float r = length(vUv - 0.5) * 2.0; gl_FragColor = vec4(uColor * pow(r, 6.0) * 0.5, 1.0); }',
}))
sheet.rotation.x = -Math.PI / 2
scanGroup.add(scanRing, sheet)
scene.add(scanGroup)

const platform = new THREE.Group()
platform.position.y = -1.42
const glow = (color, opacity) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide, toneMapped: false })
for (const [r, w, o] of [[1.15, 0.006, 0.9], [1.32, 0.02, 0.35], [1.36, 0.004, 0.7], [2.05, 0.004, 0.35]]) {
  const ring = new THREE.Mesh(new THREE.RingGeometry(r - w, r + w, 256), glow('#5fe3ff', o))
  ring.rotation.x = -Math.PI / 2
  platform.add(ring)
}
const arcs = []
for (const [r, start, len, speed] of [[1.5, 0, 1.4, 0.12], [1.5, 3.4, 0.9, 0.12], [1.72, 1.2, 2.2, -0.07], [1.9, 4.2, 0.5, 0.2]]) {
  const arc = new THREE.Mesh(new THREE.RingGeometry(r - 0.012, r + 0.012, 128, 1, start, len), glow('#8feeff', 0.75))
  arc.rotation.x = -Math.PI / 2
  arc.userData.speed = speed
  arcs.push(arc)
  platform.add(arc)
}
{
  const n = 144
  const ticks = new THREE.InstancedMesh(new THREE.PlaneGeometry(0.008, 1), glow('#5fe3ff', 0.55), n)
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2
    const len = i % 12 === 0 ? 0.12 : 0.05
    q.setFromEuler(new THREE.Euler(-Math.PI / 2, 0, -a + Math.PI / 2))
    m.compose(new THREE.Vector3(Math.cos(a) * (1.62 + len / 2), 0, Math.sin(a) * (1.62 + len / 2)), q, new THREE.Vector3(1, len, 1))
    ticks.setMatrixAt(i, m)
  }
  platform.add(ticks)
}
// Soft projection beam: brightest at the platform, fading upward, and faded
// where the wall is seen edge-on so it reads as light rather than glass.
const beam = new THREE.Mesh(new THREE.CylinderGeometry(1.0, 1.12, 1.6, 96, 1, true), new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.FrontSide,
  uniforms: { uColor: { value: CYAN } },
  vertexShader: `varying vec2 vUv; varying float vFace;
    void main(){ vUv = uv; vec4 mv = modelViewMatrix * vec4(position,1.0);
      vFace = abs(dot(normalize(normalMatrix * normal), normalize(-mv.xyz)));
      gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform vec3 uColor; varying vec2 vUv; varying float vFace;
    void main(){ gl_FragColor = vec4(uColor * pow(1.0 - vUv.y, 4.0) * pow(vFace, 1.5) * 0.1, 1.0); }`,
}))
beam.position.y = 0.8
platform.add(beam)
const grid = new THREE.GridHelper(40, 80, '#1b4a63', '#0f2738')
grid.position.y = -0.002
grid.material.transparent = true
grid.material.opacity = 0.55
platform.add(grid)
scene.add(platform)

// --- post-processing ----------------------------------------------------------
const composer = new EffectComposer(renderer)
composer.addPass(new RenderPass(scene, camera))
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.85, 0.55, 0.12)
composer.addPass(bloom)
composer.addPass(new OutputPass())

// --- HUD ----------------------------------------------------------------------
const callouts = new Callouts(document.querySelector('#callouts'), camera)
callouts.avoid([document.querySelector('.panel-left')], [document.querySelector('.panel-right')])
callouts.add({ object: embryo, local: anchors.icm, side: -1, slot: 0, className: 'warm',
  html: '<b>Inner cell mass · A</b><span>becomes the baby</span>' })
callouts.add({ object: embryo, local: anchors.cavity, side: -1, slot: 2,
  html: '<b>Blastocoel</b><span>fluid-filled cavity</span>' })
callouts.add({ object: embryo, local: anchors.te, side: 1, slot: 0,
  html: '<b>Trophectoderm · A</b><span>becomes the placenta</span>' })
callouts.add({ object: embryo, local: anchors.zona, side: 1, slot: 2,
  html: '<b>Zona pellucida</b><span>shell, thinning at grade 4</span>' })

document.querySelectorAll('#exp-seg i').forEach((el, i) => setTimeout(() => el.classList.toggle('on', i < 4), 900 + i * 140))
const C = 2 * Math.PI * 50
const setArc = (el, from, to) => {
  el.style.strokeDasharray = `${C * (to - from)} ${C}`
  el.style.strokeDashoffset = `${-C * from}`
}
const gValue = document.querySelector('#g-value')
const probEl = document.querySelector('#prob')
setArc(document.querySelector('#g-band'), 0.30, 0.53)

// --- controls -------------------------------------------------------------------
const state = { cutaway: true, rotate: true, scanStart: 0.8 }
const toggle = (id, key, fn) => document.querySelector(id).addEventListener('click', (e) => {
  state[key] = !state[key]
  e.currentTarget.classList.toggle('on', state[key])
  fn?.(state[key])
})
toggle('#btn-cut', 'cutaway')
toggle('#btn-rotate', 'rotate')
toggle('#btn-labels', 'labels', (v) => callouts.setVisible(v))
state.labels = true
document.querySelector('#btn-scan').addEventListener('click', () => { state.scanStart = clock.getElapsedTime() })

function resize() {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  composer.setSize(w, h)
  bloom.resolution.set(w, h)
  camera.aspect = w / h
  camera.updateProjectionMatrix()
}
window.addEventListener('resize', resize)
resize()

const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2)
const clock = new THREE.Clock()
const camDir = new THREE.Vector3()
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime()
  uniforms.uTime.value = t

  // intro dolly, then hand over to orbit controls
  const intro = Math.min(1, t / 2.6)
  if (intro < 1) {
    camera.position.lerpVectors(startPos, homePos, ease(intro))
    camera.lookAt(target)
  } else if (!controls.enabled) {
    controls.enabled = true
  }
  if (controls.enabled) controls.update()

  // scan sweep: bottom to top, then park out of sight
  const p = (t - state.scanStart) / 3.0
  const y = p >= 0 && p <= 1 ? THREE.MathUtils.lerp(-1.02, 1.02, ease(p)) : 99
  uniforms.uScanY.value = y
  scanGroup.visible = y < 50
  if (scanGroup.visible) {
    scanGroup.position.y = y
    const r = Math.sqrt(Math.max(0, 1.06 * 1.06 - y * y))
    scanRing.scale.setScalar(r)
    sheet.scale.setScalar(r)
  }

  // gauge fills once the scan has passed
  const g = THREE.MathUtils.clamp((t - state.scanStart - 1.2) / 1.6, 0, 1)
  setArc(gValue, 0, 0.41 * ease(g))
  probEl.textContent = Math.round(41 * ease(g))

  if (state.rotate) embryo.rotation.y += 0.0025
  arcs.forEach((a) => { a.rotation.z += a.userData.speed * 0.01 })

  // cut away the near cap so the inner cell mass is visible from any angle
  camDir.subVectors(camera.position, target).normalize()
  cut.normal.copy(camDir).negate()
  cut.constant = state.cutaway ? 0.3 : 10

  composer.render()
  callouts.update(embryo.position, 1.0)
})

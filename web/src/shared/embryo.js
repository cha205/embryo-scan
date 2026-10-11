// Procedural 3D blastocyst, driven by its Gardner grade.
//
// This is an illustration, not a reconstruction: the AI works on 2D microscope
// photos. The 3D model is built from the predicted grade so the viewer can see
// what "expansion 4, inner cell mass A, trophectoderm A" means.
//
// Parts (all children of the returned group, unit = zona outer radius):
//   zona     the shell around the embryo (zona pellucida)
//   te       trophectoderm: one layer of flat cells lining the inside (-> placenta)
//   icm      inner cell mass: a tight cluster at one side (-> the baby)
//   fluid    particles drifting in the fluid-filled cavity (blastocoel)
import * as THREE from 'three'

export function rng(seed = 1) {
  let s = seed >>> 0 || 1
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
}

function gauss(rand) {
  return Math.sqrt(-2 * Math.log(Math.max(rand(), 1e-9))) * Math.cos(2 * Math.PI * rand())
}

export function fibonacciSphere(n, rand, jitter = 0.25) {
  const pts = []
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - ((i + 0.5) / n) * 2
    const r = Math.sqrt(1 - y * y)
    const t = golden * i
    const p = new THREE.Vector3(Math.cos(t) * r, y, Math.sin(t) * r)
    p.x += (rand() - 0.5) * jitter / Math.sqrt(n)
    p.y += (rand() - 0.5) * jitter / Math.sqrt(n)
    p.z += (rand() - 0.5) * jitter / Math.sqrt(n)
    pts.push(p.normalize())
  }
  return pts
}

// What each grade looks like. Expansion: 1-2 small cavity, 3 full, 4 expanded
// with a thin zona, 5 hatching. ICM/TE: A = many tight cells, C = few loose ones.
export const GRADE = {
  expansion: {
    1: { body: 0.78, zona: 0.11, cavity: 0.35 },
    2: { body: 0.82, zona: 0.10, cavity: 0.6 },
    3: { body: 0.88, zona: 0.085, cavity: 1 },
    4: { body: 0.93, zona: 0.05, cavity: 1 },
    5: { body: 0.95, zona: 0.035, cavity: 1 },
  },
  icm: { A: { cells: 44, size: 0.072, cohesion: 1.0 }, B: { cells: 24, size: 0.082, cohesion: 0.6 }, C: { cells: 9, size: 0.095, cohesion: 0.25 } },
  te: { A: { cells: 170, irregular: 0.12 }, B: { cells: 95, irregular: 0.35 }, C: { cells: 42, irregular: 0.7 } },
}

const Z = new THREE.Vector3(0, 0, 1)

// teSize: trophectoderm cell width relative to its share of the surface
// (bigger = cells overlap into a continuous layer); teFlat: cell thickness.
export function createBlastocyst({ expansion = 4, icm = 'A', te = 'A', seed = 7, materials, icmDirection, teSize = 0.62, teFlat = 0.045 } = {}) {
  const rand = rng(seed)
  const E = GRADE.expansion[expansion]
  const group = new THREE.Group()
  group.name = 'blastocyst'
  const rT = E.body - 0.035 // trophectoderm mid-surface

  // Zona pellucida: outer + inner surface of a shell
  const zona = new THREE.Group()
  zona.name = 'zona'
  const zOuter = new THREE.Mesh(new THREE.SphereGeometry(1, 128, 96), materials.zona)
  const zInner = new THREE.Mesh(new THREE.SphereGeometry(1 - E.zona, 128, 96), materials.zonaInner || materials.zona)
  zona.add(zOuter, zInner)
  group.add(zona)

  // Trophectoderm: flattened cells tiling the inside of the shell
  const T = GRADE.te[te]
  const tePts = fibonacciSphere(T.cells, rand, 0.9)
  const cellR = (2 * rT) / Math.sqrt(T.cells)
  const teGeo = new THREE.IcosahedronGeometry(1, 3)
  const teMesh = new THREE.InstancedMesh(teGeo, materials.te, tePts.length)
  teMesh.name = 'te'
  const m = new THREE.Matrix4()
  const q = new THREE.Quaternion()
  const twist = new THREE.Quaternion()
  const s = new THREE.Vector3()
  const teCells = []
  tePts.forEach((n, i) => {
    const size = cellR * teSize * (1 + (rand() - 0.5) * T.irregular * 1.6)
    q.setFromUnitVectors(Z, n).multiply(twist.setFromAxisAngle(Z, rand() * Math.PI * 2))
    s.set(size * (1 + (rand() - 0.5) * T.irregular), size, teFlat + rand() * teFlat * 0.25)
    const p = n.clone().multiplyScalar(rT)
    m.compose(p, q, s)
    teMesh.setMatrixAt(i, m)
    teCells.push({ normal: n.clone(), position: p, quaternion: q.clone(), scale: s.clone() })
  })
  teMesh.userData.cells = teCells
  // slight per-cell shade differences read as living tissue rather than plastic
  const shade = new THREE.Color()
  for (let i = 0; i < tePts.length; i++) teMesh.setColorAt(i, shade.setScalar(0.86 + rand() * 0.14))
  group.add(teMesh)

  // Inner cell mass: relaxed cluster pressed against one side of the shell
  const I = GRADE.icm[icm]
  const dir = (icmDirection || new THREE.Vector3(-0.45, 0.62, 0.64)).clone().normalize()
  const home = dir.clone().multiplyScalar(rT - 0.16)
  const t1 = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0.01)).normalize()
  const t2 = new THREE.Vector3().crossVectors(dir, t1).normalize()
  const spread = 0.13 + 0.12 * (1 - I.cohesion)
  const pts = []
  for (let i = 0; i < I.cells; i++) {
    pts.push(home.clone()
      .addScaledVector(t1, gauss(rand) * spread)
      .addScaledVector(t2, gauss(rand) * spread)
      .addScaledVector(dir, -Math.abs(gauss(rand)) * 0.08))
  }
  const minD = I.size * 1.75
  for (let it = 0; it < 80; it++) {
    for (let a = 0; a < pts.length; a++) {
      for (let b = a + 1; b < pts.length; b++) {
        const d = pts[a].distanceTo(pts[b])
        if (d < minD && d > 1e-5) {
          const push = new THREE.Vector3().subVectors(pts[a], pts[b]).multiplyScalar((minD - d) / d * 0.5)
          pts[a].add(push)
          pts[b].sub(push)
        }
      }
      pts[a].lerp(home, 0.02 * I.cohesion)
      const lim = rT - 0.03 - I.size
      if (pts[a].length() > lim) pts[a].setLength(lim)
    }
  }
  const icmMesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 3), materials.icm, pts.length)
  icmMesh.name = 'icm'
  pts.forEach((p, i) => {
    const r = I.size * (0.85 + rand() * 0.35)
    q.setFromEuler(new THREE.Euler(rand() * 6, rand() * 6, rand() * 6))
    m.compose(p, q, s.set(r * (0.9 + rand() * 0.2), r, r * (0.85 + rand() * 0.2)))
    icmMesh.setMatrixAt(i, m)
    icmMesh.setColorAt(i, shade.setScalar(0.8 + rand() * 0.2))
  })
  group.add(icmMesh)
  const icmCenter = pts.reduce((acc, p) => acc.add(p), new THREE.Vector3()).divideScalar(pts.length)

  // Fluid: soft particles in the cavity
  let fluid = null
  if (materials.fluid) {
    const n = 700
    const pos = new Float32Array(n * 3)
    const seedAttr = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      const v = new THREE.Vector3(gauss(rand), gauss(rand), gauss(rand)).normalize().multiplyScalar(Math.cbrt(rand()) * rT * 0.86)
      pos.set([v.x, v.y, v.z], i * 3)
      seedAttr[i] = rand()
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('seed', new THREE.BufferAttribute(seedAttr, 1))
    fluid = new THREE.Points(g, materials.fluid)
    fluid.name = 'fluid'
    group.add(fluid)
  }

  group.userData = {
    grade: `${expansion}${icm}${te}`,
    parts: { zona, te: teMesh, icm: icmMesh, fluid },
    anchors: {
      icm: icmCenter,
      te: dir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), 2.3).multiplyScalar(rT),
      zona: new THREE.Vector3(0.55, -0.62, 0.56).normalize(),
      cavity: dir.clone().multiplyScalar(-0.35),
    },
    teRadius: rT,
  }
  return group
}

// Inject a view-angle rim glow and a moving horizontal "scan band" into any
// MeshStandard/MeshPhysical material (works with InstancedMesh).
export function patchHolo(material, uniforms, { rim = '#5fe3ff', power = 2.2, strength = 1.0, scan = '#b9f6ff', scanStrength = 1.6 } = {}) {
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms, {
      uRim: { value: new THREE.Color(rim) },
      uRimPower: { value: power },
      uRimStrength: { value: strength },
      uScanColor: { value: new THREE.Color(scan) },
      uScanStrength: { value: scanStrength },
    })
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vScanWorld;')
      .replace('#include <project_vertex>', `#include <project_vertex>
        vec4 scanWP = vec4(transformed, 1.0);
        #ifdef USE_INSTANCING
          scanWP = instanceMatrix * scanWP;
        #endif
        vScanWorld = (modelMatrix * scanWP).xyz;`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vScanWorld;
        uniform float uScanY, uRimPower, uRimStrength, uScanStrength;
        uniform vec3 uRim, uScanColor;`)
      .replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>
        float fres = pow(1.0 - abs(dot(normalize(normal), normalize(vViewPosition))), uRimPower);
        totalEmissiveRadiance += uRim * fres * uRimStrength;
        float band = exp(-pow((vScanWorld.y - uScanY) / 0.03, 2.0));
        totalEmissiveRadiance += uScanColor * band * uScanStrength;`)
  }
  material.customProgramCacheKey = () => `holo-${rim}-${power}-${strength}-${scan}`
  return material
}

// The trophectoderm as one continuous sheet: a 3D Voronoi (Worley) pattern on
// a sphere draws cell borders, a nucleus near each cell centre, and a slight
// per-cell shade, instead of hundreds of separate discs.
export function epitheliumMaterial({ freq = 7.5, nucleus = '#7186c2', ...params } = {}) {
  const mat = new THREE.MeshPhysicalMaterial(params)
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uFreq = { value: freq }
    shader.uniforms.uNucleus = { value: new THREE.Color(nucleus) }
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vObj;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvObj = position;')
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        varying vec3 vObj;
        uniform float uFreq;
        uniform vec3 uNucleus;
        vec3 hash3(vec3 p) {
          p = vec3(dot(p, vec3(127.1, 311.7, 74.7)), dot(p, vec3(269.5, 183.3, 246.1)), dot(p, vec3(113.5, 271.9, 124.6)));
          return fract(sin(p) * 43758.5453123);
        }
        vec3 worley(vec3 x) {
          vec3 p = floor(x), f = fract(x);
          float d1 = 8.0, d2 = 8.0, id = 0.0;
          for (int k = -1; k <= 1; k++) for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
            vec3 b = vec3(float(i), float(j), float(k));
            vec3 r = b + hash3(p + b) - f;
            float d = dot(r, r);
            if (d < d1) { d2 = d1; d1 = d; id = dot(p + b, vec3(7.0, 157.0, 113.0)); }
            else if (d < d2) { d2 = d; }
          }
          return vec3(sqrt(d1), sqrt(d2), fract(sin(id) * 43758.5453));
        }`)
      .replace('#include <color_fragment>', `#include <color_fragment>
        vec3 w = worley(normalize(vObj) * uFreq);
        float border = 1.0 - smoothstep(0.015, 0.075, w.y - w.x);
        float nuc = 1.0 - smoothstep(0.11, 0.17, w.x);
        diffuseColor.rgb *= (0.9 + 0.1 * w.z) * mix(1.0, 0.72, border);
        diffuseColor.rgb = mix(diffuseColor.rgb, uNucleus, nuc * 0.7);`)
  }
  mat.customProgramCacheKey = () => `epithelium-${freq}`
  return mat
}

export function fluidMaterial(uniforms, color = '#9fe8ff', size = 2.2, opacity = 0.55) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { ...uniforms, uColor: { value: new THREE.Color(color) }, uSize: { value: size }, uOpacity: { value: opacity } },
    vertexShader: /* glsl */`
      attribute float seed;
      uniform float uTime, uSize;
      varying float vTw;
      void main() {
        vec3 p = position;
        p += 0.02 * vec3(sin(uTime * 0.4 + seed * 40.0), cos(uTime * 0.33 + seed * 31.0), sin(uTime * 0.27 + seed * 17.0));
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        vTw = 0.55 + 0.45 * sin(uTime * 1.3 + seed * 60.0);
        gl_PointSize = uSize * (0.6 + seed) * (6.0 / -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */`
      uniform vec3 uColor;
      uniform float uOpacity;
      varying float vTw;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float a = smoothstep(0.5, 0.0, d);
        gl_FragColor = vec4(uColor, a * uOpacity * vTw);
      }`,
  })
}

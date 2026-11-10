// Held pigment chunks (design §5): a dropped chunk is not let go into the flow all at once. Its
// particles are held (kinematic: flags bit 0, and bit 2 marks them as a chunk's) and the lump sinks
// at a slow, steady speed through the level it was set down on; a particle is let go as it crosses
// that level, so the chunk feeds in from the bottom as a thin stream over a few revolutions of the
// roll and its pigment is laid around the whole circumference instead of going through the nip as
// one patch (measured: a chunk let go whole becomes a patch a third to a half of the circumference
// long, which then circles for many revolutions without spreading).
//
// The level starts where the chunk was set down (the top of the material under it, which the host
// probes to place it) and is not found by contact particle by particle: particles just let go
// raise the grid's mass under the lump, and a release on touching material set off the layer above,
// a chain reaction that fed most of a chunk in within a second. With a level the feed is even.
// The level only ever moves down: `settle_` (one invocation a frame) looks for material just under
// each chunk (the node mass the grid update leaves, at the middle of its footprint and four points
// around it) and, where there is none, lowers the level and the lump with it at CHUNK_FALL, so a
// chunk follows the bank down as it drains instead of hanging in the air above it. If the bank
// rises the chunk is partly buried and keeps feeding at its level.
//
// `step_` runs once a frame over the particles; all particles of a chunk move alike, so it keeps
// its shape. `drop_` lets every held particle go at once: the host runs it when an operator move
// starts or the feed is switched off.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flags : array<u32>;
// the chunk table: [0] = how many chunks; then 8 words a chunk from word 4: first particle,
// particle count, and as float bits the level it feeds in at (y), how far settle_ lowered it this
// frame, its footprint's centre x and z and its radius
@group(0) @binding(4) var<storage, read_write> chunks : array<u32>;
@group(0) @binding(5) var<storage, read> gvel : array<vec4<f32>>;   // xyz velocity, w the node's mass (grid.wgsl)

const CHUNK_FALL : f32 = 0.6;    // sim units / s: how fast a chunk with nothing under it comes down

/// Node mass at a world point (0 outside the grid).
fn massAt(x : vec3<f32>) -> f32 {
  let c = vec3<i32>(round(x * P.hdt.y));
  if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(P.grid.xyz))) { return 0.0; }
  return gvel[nodeIndexI(c)].w;
}

@compute @workgroup_size(1)
fn settle_() {
  let n = min(chunks[0], 32u);
  let h = P.hdt.x;
  for (var i = 0u; i < n; i++) {
    let w = 4u + 8u * i;
    let level = bitcast<f32>(chunks[w + 2u]);
    let cx = bitcast<f32>(chunks[w + 4u]);
    let cz = bitcast<f32>(chunks[w + 5u]);
    let d = 0.5 * bitcast<f32>(chunks[w + 6u]);
    let y = level - 0.75 * h;
    let m = max(massAt(vec3<f32>(cx, y, cz)), max(
      max(massAt(vec3<f32>(cx - d, y, cz)), massAt(vec3<f32>(cx + d, y, cz))),
      max(massAt(vec3<f32>(cx, y, cz - d)), massAt(vec3<f32>(cx, y, cz + d)))));
    var fall = 0.0;
    if (m < P.chunk.y && level > 3.0 * h) { fall = CHUNK_FALL * P.domain.w; }
    chunks[w + 2u] = bitcast<u32>(level - fall);
    chunks[w + 3u] = bitcast<u32>(fall);
  }
}

@compute @workgroup_size(128)
fn step_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let fl = flags[p];
  if ((fl & 4u) == 0u) { return; }
  // let go by something else (a reset of the flags): no longer a chunk's
  if ((fl & 1u) == 0u) { flags[p] = fl & ~4u; return; }
  // which chunk
  let n = min(chunks[0], 32u);
  var level = -1.0;
  var fall = 0.0;
  for (var i = 0u; i < n; i++) {
    let w = 4u + 8u * i;
    if (p >= chunks[w] && p < chunks[w] + chunks[w + 1u]) {
      level = bitcast<f32>(chunks[w + 2u]);
      fall = bitcast<f32>(chunks[w + 3u]);
      break;
    }
  }
  if (level < 0.0) { flags[p] = 0u; return; }
  let speed = P.chunk.x;
  var x = pos[p].xyz;
  x.y = max(x.y - speed * P.domain.w - fall, 1.5 * P.hdt.x);
  pos[p] = vec4<f32>(x, pos[p].w);
  if (x.y < level) {
    // it has reached the bank: let it go where it is, moving as the lump was
    flags[p] = 0u;
    vel[p] = vec4<f32>(0.0, -speed, 0.0, 0.0);
  }
}

@compute @workgroup_size(128)
fn drop_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let fl = flags[p];
  if ((fl & 4u) == 0u) { return; }
  if ((fl & 1u) != 0u) { vel[p] = vec4<f32>(0.0); }
  flags[p] = fl & ~5u;
}

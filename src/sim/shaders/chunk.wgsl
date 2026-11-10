// Held pigment chunks (design §5): a dropped chunk is not let go into the flow all at once. Its
// particles are held (kinematic: flags bit 0, and bit 2 marks them as a chunk's) and the lump sinks
// into the bank at a slow, steady speed; a particle is let go when it reaches material, so the
// chunk feeds in from the bottom as a thin stream over a few revolutions of the roll and its
// pigment is laid around the whole circumference instead of going through the nip as one patch
// (measured: a chunk let go whole becomes a patch a third to a half of the circumference long,
// which then circles for many revolutions without spreading).
//
// `step_` runs once a frame over the particles. A held particle sinks at the feed speed while its
// chunk is in touch with material and falls at CHUNK_FALL while it is not, so the lump follows
// the bank's surface as it rises and drops; all particles of a chunk move alike, so it keeps its
// shape. In touch means some particle of the chunk reached material within the last P.chunk.z
// frames: at the feed speed the next layer of particles takes many frames to arrive, and a chunk
// that fell whenever one frame passed without a touch would feed in at the falling speed. It touches where the solver grid carries mass at its node (the node mass the grid update
// leaves beside the velocity; held particles scatter none, so that is free material only).
// `end_` (one invocation) turns this frame's touch counts into next frame's. `drop_` lets every
// held particle go at once: the host runs it when an operator move starts or the feed is switched
// off.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flags : array<u32>;
@group(0) @binding(4) var<storage, read> gvel : array<vec4<f32>>;   // xyz velocity, w the node's mass (grid.wgsl)
// the chunk table: [0] = how many chunks; then 4 words a chunk from word 4: first particle,
// particle count, particles that touched material this frame, frames left of being in touch
@group(0) @binding(5) var<storage, read_write> chunks : array<atomic<u32>>;

const CHUNK_HELD : u32 = 5u;     // flags of a held chunk particle: kinematic (bit 0) and a chunk's (bit 2)
const CHUNK_FALL : f32 = 0.6;    // sim units / s: how fast a chunk that touches nothing comes down

@compute @workgroup_size(128)
fn step_(@builtin(global_invocation_id) gid : vec3<u32>, @builtin(num_workgroups) nwg : vec3<u32>) {
  let p = particleIndex(gid, nwg);
  if (p >= P.grid.w) { return; }
  let fl = flags[p];
  if ((fl & 4u) == 0u) { return; }
  // let go by something else (a reset of the flags): no longer a chunk's
  if ((fl & 1u) == 0u) { flags[p] = fl & ~4u; return; }
  // which chunk
  let n = min(atomicLoad(&chunks[0]), 32u);
  var at = 0u;
  var found = false;
  for (var i = 0u; i < n; i++) {
    let w = 4u + 4u * i;
    let first = atomicLoad(&chunks[w]);
    if (p >= first && p < first + atomicLoad(&chunks[w + 1u])) { at = w; found = true; break; }
  }
  if (!found) { flags[p] = 0u; return; }
  let touching = atomicLoad(&chunks[at + 3u]) > 0u;
  let speed = select(CHUNK_FALL, P.chunk.x, touching);
  var x = pos[p].xyz;
  x.y = max(x.y - speed * P.domain.w, 1.5 * P.hdt.x);
  pos[p] = vec4<f32>(x, pos[p].w);
  let c = clamp(vec3<i32>(round(x * P.hdt.y)), vec3<i32>(0), vec3<i32>(P.grid.xyz) - vec3<i32>(1));
  if (gvel[nodeIndexI(c)].w >= P.chunk.y) {
    // it has reached the bank: let it go where it is, moving as the lump was
    atomicAdd(&chunks[at + 2u], 1u);
    flags[p] = 0u;
    vel[p] = vec4<f32>(0.0, -speed, 0.0, 0.0);
  }
}

@compute @workgroup_size(1)
fn end_() {
  let n = min(atomicLoad(&chunks[0]), 32u);
  for (var i = 0u; i < n; i++) {
    let w = 4u + 4u * i;
    var left = atomicLoad(&chunks[w + 3u]);
    if (atomicLoad(&chunks[w + 2u]) > 0u) { left = u32(P.chunk.z); } else if (left > 0u) { left -= 1u; }
    atomicStore(&chunks[w + 3u], left);
    atomicStore(&chunks[w + 2u], 0u);
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

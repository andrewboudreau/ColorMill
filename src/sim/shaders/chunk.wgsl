// Held pigment chunks (design §5): a dropped chunk is not let go into the flow all at once. Its
// particles are held (kinematic: flags bit 0, and bit 2 marks them as a chunk's) and the lump sinks
// at a slow, steady speed through the level it was set down on; a particle is let go as it crosses
// that level, so the chunk feeds in from the bottom as a thin stream over a few revolutions of the
// roll and its pigment is laid around the whole circumference instead of going through the nip as
// one patch (measured: a chunk let go whole becomes a patch a third to a half of the circumference
// long, which then circles for many revolutions without spreading).
//
// The level is fixed when the chunk is set down (the top of the material under it, which the host
// probes to place it), not found by contact each frame: particles just let go raise the grid's mass
// under the lump, and a release on touching material set off the layer above, a chain reaction
// that fed most of a chunk in within a second. With a fixed level the feed is exactly even. If the
// bank's surface has moved since, the stream falls a little way onto it or comes out just inside it.
//
// `step_` runs once a frame over the particles; all particles of a chunk move alike, so it keeps
// its shape. `drop_` lets every held particle go at once: the host runs it when an operator move
// starts or the feed is switched off.
@group(0) @binding(1) var<storage, read_write> pos : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> flags : array<u32>;
// the chunk table: [0] = how many chunks; then 4 words a chunk from word 4: first particle,
// particle count, the level it feeds in at (the bits of a float y), 0
@group(0) @binding(4) var<storage, read> chunks : array<u32>;

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
  for (var i = 0u; i < n; i++) {
    let w = 4u + 4u * i;
    if (p >= chunks[w] && p < chunks[w] + chunks[w + 1u]) { level = bitcast<f32>(chunks[w + 2u]); break; }
  }
  if (level < 0.0) { flags[p] = 0u; return; }
  let speed = P.chunk.x;
  var x = pos[p].xyz;
  x.y = max(x.y - speed * P.domain.w, 1.5 * P.hdt.x);
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

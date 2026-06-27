# Vendored: Mixbox

Mixbox is a pigment-mixing library based on the Kubelka&ndash;Munk model.
Blue + yellow mixes to green, the way real paint does, instead of the gray you
get from linear RGB averaging.

- **Authors:** Šárka Sochorová and Ondřej Jamriška (Secret Weapons)
- **Paper:** *Practical Pigment Mixing for Digital Painting*, SIGGRAPH Asia 2021
- **Upstream:** https://github.com/scrtwpns/mixbox
- **License:** Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0).
  Non-commercial use only. For a commercial license contact `mixbox@scrtwpns.com`.

## Files

- `mixbox.js` — official UMD build, fetched from
  `scrtwpns/mixbox/master/javascript/mixbox.js`. **Self-contained:** it embeds
  and decodes its own 512×512 LUT, so no separate image is needed for the
  JS/WebGL path. Exposes `mixbox.lerp`, `mixbox.glsl()`, `mixbox.lutTexture(gl)`,
  and the latent-space helpers.
- `mixbox.glsl` — the GLSL source (kept as readable reference; `mixbox.glsl()`
  in `mixbox.js` returns the same thing for injection into a shader).

The 4096×4096 `mixbox_lut.png` distributed for the C/C++/native bindings is
**not** vendored here — it is an *encoded* LUT that requires the upstream
decoder, and the JS build already carries its own decoded copy.

## Usage in a WebGL2 shader

```js
var lut = mixbox.lutTexture(gl);            // upload decoded 512x512 LUT
gl.bindTexture(gl.TEXTURE_2D, lut);
// fragment shader = header + mixbox.glsl() + main(); declare `uniform sampler2D mixbox_lut;`
// then call: vec3 c = mixbox_lerp(a, b, t);
```

See `../../pigment.html` for a working proof of concept.

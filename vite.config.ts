import { defineConfig } from 'vitest/config';

export default defineConfig({
  base: '/ColorMill/',
  build: {
    target: 'es2022'
  },
  test: {
    globals: true,
    include: ['tests/**/*.test.ts']
  }
});

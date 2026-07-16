import { converter } from 'culori';
import fs from 'fs';

const toOklch = converter('oklch');
const css = fs.readFileSync('src/index.css', 'utf-8');

const modified = css.replace(/(--[a-zA-Z0-9-]+):\s*[^;]+;\s*\/\*.*?#([0-9A-Fa-f]{6}).*?\*\//g, (match, varName, hex) => {
  const res = toOklch('#' + hex);
  const l = res.l.toFixed(3).replace(/\.000$/, '');
  const c = res.c.toFixed(3).replace(/\.000$/, '');
  const h = (res.h || 0).toFixed(1).replace(/\.0$/, '');
  // keep the original comment but replace the value
  const comment = match.match(/\/\*.*\*\//)[0];
  return `${varName}: ${l} ${c} ${h}; ${comment}`;
});

// also need to replace the @theme vars from hsl() to oklch()
const themeModified = modified.replace(/hsl\(var\((--[a-zA-Z0-9-]+)\)\)/g, 'oklch(var($1))');
// and replace the body background from hsl to oklch
const bodyModified = themeModified.replace(/background-color: hsl\(var\(--background\)\);/, 'background-color: oklch(var(--background));')
    .replace(/color: hsl\(var\(--foreground\)\);/, 'color: oklch(var(--foreground));')
    .replace(/::-webkit-scrollbar-track \{ background: hsl\(var\(--background\)\); \}/, '::-webkit-scrollbar-track { background: oklch(var(--background)); }')
    .replace(/::-webkit-scrollbar-thumb \{ background: hsl\(var\(--border\)\); border-radius: 10px; \}/, '::-webkit-scrollbar-thumb { background: oklch(var(--border)); border-radius: 10px; }');

fs.writeFileSync('src/index.css', bodyModified);
console.log('Done');

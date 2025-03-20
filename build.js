import fs from 'fs';
import { execSync } from 'child_process';

// Run TypeScript compiler
console.log('Running TypeScript compiler...');
execSync('tsc', { stdio: 'inherit' });

// Make the output file executable
console.log('Making index.js executable...');
try {
  fs.chmodSync('build/index.js', '755');
  console.log('Build completed successfully!');
} catch (error) {
  console.error('Error making file executable:', error);
  process.exit(1);
}

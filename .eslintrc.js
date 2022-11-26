module.exports = {
  root: true,
  parser: 'babel-eslint',
  parserOptions: {
    sourceType: 'module',
  },
  env: {
    browser: true,
    node: true,
  },
  extends: 'airbnb-base',
  globals: {
    __static: true,
  },
  plugins: ['html'],
  rules: {
    'no-console': 'off',
    'global-require': 0,
    'import/no-unresolved': 0,
    'no-param-reassign': 0,
    'no-shadow': 0,
    'import/extensions': 0,
    'import/newline-after-import': 0,
    'no-multi-assign': 0,
    // allow debugger during development
    'no-debugger': process.env.NODE_ENV === 'production' ? 2 : 0,
    'class-methods-use-this': 'off',
    'linebreak-style': [0, 'error', 'windows'],
    'no-plusplus': 'off',
    'prefer-destructuring': 'off',
    'no-underscore-dangle': 'off',
    'import/no-extraneous-dependencies': ['error', { devDependencies: true }],
    'arrow-parens': 0,
    camelcase: 0,
    'no-unused-expressions': 0,
  },
};

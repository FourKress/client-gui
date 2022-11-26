module.exports = function (api) {
  api.cache(true);

  const presets = [
    [
      '@babel/preset-env',
      {
        modules: false,
        targets: {
          browsers: ['> 1%', 'last 5 versions', 'ie >= 11'],
        },
        useBuiltIns: 'usage',
        corejs: 3,
      },
    ],
    '@vue/babel-preset-jsx',
  ];
  const plugins = [
    '@babel/plugin-transform-runtime',
    [
      '@babel/plugin-proposal-class-properties',
      {
        loose: false,
      },
    ],
    [
      '@babel/plugin-proposal-decorators',
      {
        legacy: true,
      },
    ],
    '@babel/plugin-transform-modules-commonjs',
  ];

  return {
    presets,
    plugins,
  };
};

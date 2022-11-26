<template>
  <div id="app">
    <router-view></router-view>
  </div>
</template>

<script>
import config from '../../package.json';

export default {
  name: 'App',
  components: {
  },
  data() {
    return {
      isLogin: false,
    };
  },
  async created() {
    document.title = `风电场优化软件 V${config.version}`;
  },
  methods: {

  },
};
</script>

<style lang="less">
* {
  margin: 0;
  padding: 0;
}
html, body {
  width: 100%;
  height: 100%;
}
#app {
  height: 100%;
}
</style>

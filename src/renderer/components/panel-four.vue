<template>
  <div class="tab-panel">
    <el-tabs tab-position="left">
      <el-tab-pane label="图1+图2">
        <ImgPanel />
      </el-tab-pane>
      <el-tab-pane label="图3+图4">
        <ImgPanel />
      </el-tab-pane>
      <el-tab-pane label="图5+图6">
        <ImgPanel />
      </el-tab-pane>
    </el-tabs>
    <div>
      <el-button type="primary" @click="onUpdate" class="save-btn">
        刷新
      </el-button>
    </div>
  </div>
</template>

<script>
import ImgPanel from './img-panel';

export default {
  name: 'panel-four',
  components: {
    ImgPanel,
  },
  methods: {
    onUpdate() {},
  },
};
</script>

<style scoped lang="less">
.tab-panel {
  height: 100%;
  display: flex;
  flex-direction: column;

  .el-tabs--left {
    flex: 1;

    /deep/ .el-tabs__header {
      margin-right: 15px;
    }
  }

  /deep/ .el-tabs__item {
    padding:  0 15px 0 0 !important;
    color: #606266;
    &.is-active {
      color: #409EFF;
    }
  }
  .el-tab-pane {
    color: #606266;
  }

  .save-btn {
    width: 120px;
    margin: 30px auto 0;
    display: block;
  }
}
</style>

<template>
  <div class="tab-panel">
    <el-tabs tab-position="left">
      <el-tab-pane label="图1+图2">
        <ImgPanel :urlArr="first" :srcList="srcList" />
        <div class="tips">
          <span>显示风机的性能曲线</span>
        </div>
      </el-tab-pane>
      <el-tab-pane label="图3+图4">
        <ImgPanel :urlArr="second" :srcList="srcList" />
        <div class="tips">
          <span>动态监视计算过程</span>
          <span>动态监视优化进展，当该线平直意味着不在更新优化，可以中断</span>
        </div>
      </el-tab-pane>
      <el-tab-pane label="图5+图6">
        <ImgPanel :urlArr="third" :srcList="srcList" />
        <div class="tips">
          <span>显示尾流情况</span>
          <span></span>
        </div>
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
import { ipcRenderer } from 'electron';
import ImgPanel from './img-panel';

export default {
  name: 'panel-four',
  components: {
    ImgPanel,
  },
  data() {
    return {
      first: [],
      second: [],
      third: [],
      srcList: [],
    };
  },
  methods: {
    onUpdate() {
      ipcRenderer.send('getResult');
      ipcRenderer.once('result', (event, data) => {
        if (!data) return;
        this.first = [data.CT_vs_Speeds, data.Power_vs_Speeds];
        this.second = [
          data.turbines_boundary_flowfield_all_direction,
          data.iteration_status,
        ];
        this.third = [
          data.streamlines_3D_all_directions,
          data.isoheight_wind_with_reduce_all_direction_probability,
        ];
        this.srcList = [...this.first, ...this.second, ...this.third].filter(d => d);
        console.log(this.first);
      });
    },
  },
};
</script>

<style scoped lang="less">
.tab-panel {
  height: 100%;
  display: flex;
  flex-direction: column;

  .tips {
    width: 100%;
    height: 22px;
    line-height: 22px;
    display: flex;
    align-items: center;
    justify-content: center;

    span {
      flex: 1;
      text-align: center;
      color: #333;
      font-size: 12px;
    }
  }

  .el-tabs--left {
    flex: 1;

    /deep/ .el-tabs__header {
      margin-right: 15px;
    }
  }

  /deep/ .el-tabs__item {
    padding: 0 15px 0 0 !important;
    color: #606266;
    &.is-active {
      color: #409eff;
    }
  }
  .el-tab-pane {
    color: #606266;
  }

  .save-btn {
    width: 120px;
    margin: 10px auto 0;
    display: block;
  }
}
</style>

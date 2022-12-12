<template>
  <div class="img-panel">
    <div class="left">
      <el-image :src="urlArr[0]" :preview-src-list="srcList" fit="contain">
        <span slot="error" class="image-slot">暂无数据</span>
      </el-image>
    </div>
    <div class="right">
      <el-image :src="urlArr[1]" :preview-src-list="srcList" fit="contain">
        <span slot="error" class="image-slot">暂无数据</span>
      </el-image>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImgPanel',
  props: {
    urlArr: {
      type: Array,
      default: () => [],
    },
    srcList: {
      type: Array,
      default: () => [],
    },
  },
};
</script>

<style scoped lang="less">
.img-panel {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;

  img {
    width: 100%;
    height: 100%;
    display: block;
  }

  .left,
  .right {
    flex: 1;
    width: 50%;
    height: 100%;
    box-sizing: border-box;
    border: 1px solid #ccc;
  }

  .left {
    margin-right: 7.5px;
  }
  .right {
    margin-left: 7.5px;
  }
}
</style>

<style lang="less">
.el-image {
  height: 100%;
}
</style>

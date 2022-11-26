<template>
  <div class="img-panel">
    <div class="left">
      <el-image src="" fit="contain"></el-image>
    </div>
    <div class="right">
      <el-image src="" fit="contain"></el-image>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ImgPanel',
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
    background-color: #ccc;
  }

  .left {
    margin-right: 7.5px;
  }
  .right {
    margin-left: 7.5px;
  }
}
</style>

<template>
  <div class="main">
    <el-tabs type="border-card" class="tabs">
      <el-tab-pane :label="tabs[0]">
        <PanelFirst ref="PanelFirst" :historyConfig="historyConfig" :isStart="isStart" />
      </el-tab-pane>
      <el-tab-pane :label="tabs[1]">
        <PanelSecond ref="PanelSecond" :historyConfig="historyConfig" :isStart="isStart" />
      </el-tab-pane>
      <el-tab-pane :label="tabs[2]">
        <PanelThird
          ref="PanelThird"
          @start="handleStart"
          @stop="handleStop"
          :historyConfig="historyConfig"
          :isStart="isStart"
        />
      </el-tab-pane>
      <el-tab-pane :label="tabs[3]">
        <PanelFour ref="PanelFour" :isStart="isStart" />
      </el-tab-pane>
    </el-tabs>

    <div class="dynamic-output">
      <el-input
        type="textarea"
        autosize
        resize="none"
        disabled
        v-for="(msg, index) in msgList"
        :key="index"
        :value="msg"
      ></el-input>
    </div>

    <el-button
      type="text"
      :disabled="isStart"
      class="import-btn"
      @click="dialogVisible = true"
    >
      导入历史配置
    </el-button>
    <el-dialog
      title="导入历史配置"
      :visible.sync="dialogVisible"
      :close-on-press-escape="false"
      :close-on-click-modal="false"
      @close="closeDialog"
    >
      <el-form ref="form" :model="form">
        <el-form-item
          label="配置文件"
          label-width="150"
          prop="historyConfig"
          :rules="[
            {
              required: true,
              message: '请选择',
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-button type="primary" @click="onSelectOnly('historyConfig')">
            选择
          </el-button>
          <el-tag
            v-if="form.historyConfig"
            closable
            @close="handlePathCloseOnly('historyConfig')"
            type="success"
          >
            {{ form.historyConfig }}
          </el-tag>
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button @click="closeDialog">取 消</el-button>
        <el-button type="primary" @click="handleImportHistoryConfig">
          确 定
        </el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
import { ipcRenderer } from 'electron';
import PanelFirst from '../components/panel-first';
import PanelSecond from '../components/panel-second';
import PanelThird from '../components/panel-third';
import PanelFour from '../components/panel-four';
import Mixins from '../mixins';

export default {
  name: 'home',
  components: {
    PanelFirst,
    PanelSecond,
    PanelThird,
    PanelFour,
  },
  mixins: [Mixins],
  data() {
    return {
      startCount: 0,
      tabs: ['风机设置', '计算域设置', '优化器设置', '图形显示'],
      msgList: [],
      isStart: false,
      dialogVisible: false,
      form: {
        historyConfig: '',
      },
      historyConfig: null,
    };
  },
  watch: {
    msgList(val) {
      if (val.length > 200) {
        this.msgList.splice(0, 1);
      }
    },
  },
  mounted() {
  },
  methods: {
    closeDialog() {
      this.$refs.form.resetFields();
      this.$refs.form.clearValidate();
      this.dialogVisible = false;
    },
    handleImportHistoryConfig() {
      this.$refs.form.validate((valid) => {
        if (valid) {
          const loading = this.$loading({
            lock: true,
            text: '配置导入中...',
            spinner: 'el-icon-loading',
            background: 'rgba(0, 0, 0, 0.7)',
          });

          ipcRenderer.send('loadConfig', this.form.historyConfig);
          ipcRenderer.once('resultConfig', (_event, data) => {
            const config = `${data.replace(/[\r\n]/g, '').replace(/'/g, '"').replace(/False/g, 'false').replace(/True/g, 'true')}`;
            this.historyConfig = JSON.parse(config);
            console.log(this.historyConfig);
            loading.close();
            this.$message.success('历史配置导入成功');
          });
          this.closeDialog();
        }
      });
    },
    handleStart(isContinue) {
      console.log(isContinue);
      Promise.all([
        this.$refs.PanelFirst.validate(),
        this.$refs.PanelSecond.validate(),
        this.$refs.PanelThird.validate(),
      ]).then((values) => {
        console.log(values);
        const errorIndex = values.findIndex((d) => !d.valid);
        if (errorIndex !== -1) {
          this.$message.error(`请完善 ${this.tabs[errorIndex]} 的配置项`);
          return;
        }

        let formAll = {};
        values.forEach((v) => {
          formAll = {
            ...formAll,
            ...v.form,
          };
        });
        const params = {
          flag_optimizer_status: isContinue ? 'continue' : 'initialize',
        };
        Object.keys(formAll).forEach((k) => {
          if (formAll[k] !== undefined) {
            params[k] = formAll[k];
          }
        });
        console.log(params);

        this.onStart(params);
      });
    },
    onStart(params) {
      this.msgList = [];
      this.isStart = true;
      this.startCount++;
      ipcRenderer.send('start', [this.startCount, params]);

      ipcRenderer.on(`stdout_${this.startCount}`, (event, data) => {
        console.log(event, data);
        this.msgList.push(data);
      });

      ipcRenderer.on(`stderr_${this.startCount}`, (event, data) => {
        console.log(event, data);
        this.msgList.push(data);
      });

      ipcRenderer.once(`close_${this.startCount}`, (event, data) => {
        console.log(event, data);
        this.msgList.push(data);
        if (data.includes('退出码')) {
          this.isStart = false;
        }
      });
    },
    handleStop() {
      this.$confirm('确定停止计算吗?', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      })
        .then(() => {
          ipcRenderer.send('stop');
        })
        .catch(() => {});
    },
    //
    // onUpdate() {
    //   ipcRenderer.send('getResult', [this.form.filePath]);
    //   ipcRenderer.once('result', (event, data) => {
    //     console.log(event, data);
    //     this.imgUrl = data;
    //   });
    // },
  },
};
</script>

<style scoped lang="less">
.main {
  width: 100%;
  height: 100%;
  box-sizing: border-box;

  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;

  .tabs {
    width: 100%;
    box-sizing: border-box;
    flex: 1;
    display: flex;
    flex-direction: column;

    /deep/ .el-tabs__content {
      flex: 1;
      height: 100%;

      .el-tab-pane {
        height: 100%;
      }
    }
  }

  .dynamic-output {
    width: 100%;
    min-height: 300px;
    max-height: 300px;
    overflow-y: auto;
    box-sizing: border-box;
    border: 16px solid #e4e7ed;

    background-color: #fff;
    color: #333;
  }

  .import-btn {
    position: fixed;
    right: 12px;
    top: 4px;
  }
}
</style>

<style>
.el-tag {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin: 6px 0;
}

.el-textarea.is-disabled .el-textarea__inner {
  background-color: unset;
  color: unset;
  cursor: default;
  border: none;
  padding: 0 6px;
}
</style>

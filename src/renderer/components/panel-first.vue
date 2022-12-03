<template>
  <div class="tab-panel">
    <el-form
      class="form-panel"
      ref="form"
      :model="form"
      :rules="rules"
      label-width="150px"
    >
      <div class="left">
        <el-form-item label="parameters_turbine" prop="parameters_turbine">
          <el-button
            type="primary"
            :disabled="isStart || true"
            @click="onSelect('parameters_turbine')"
          >
            选择
          </el-button>
          <div class="tag-panel" v-if="form.parameters_turbine.length">
            <el-tag
              v-for="path in form.parameters_turbine"
              :key="path"
              :closable="!isStart"
              @close="handlePathClose(path, 'parameters_turbine')"
            >
              {{ path }}
            </el-tag>
          </div>
        </el-form-item>

        <el-form-item label="num_turbines" prop="num_turbines">
          <el-input :disabled="isStart" v-model="form.num_turbines"></el-input>
        </el-form-item>
        <el-form-item label="dist_threshold" prop="dist_threshold">
          <el-input
            :disabled="isStart"
            v-model="form.dist_threshold"
          ></el-input>
        </el-form-item>
        <el-form-item label="turbine_setting" prop="turbine_setting">
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('turbine_setting')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.turbine_setting"
            :closable="!isStart"
            @close="handlePathCloseOnly('turbine_setting')"
          >
            {{ form.turbine_setting }}
          </el-tag>
        </el-form-item>
      </div>
      <div class="right">
        <el-form-item
          label="is_specify_loc_turbines_initial"
          prop="is_specify_loc_turbines_initial"
          label-width="200px"
        >
          <el-checkbox
            :disabled="isStart"
            v-model="form.is_specify_loc_turbines_initial"
          ></el-checkbox>
        </el-form-item>

        <el-form-item
          label="dir_turbine_loc"
          prop="dir_turbine_loc"
          label-width="200px"
          :rules="[
            {
              required: form.is_specify_loc_turbines_initial,
              message: '请选择',
              trigger: ['blur', 'change'],
            },
            {
              validator: this.validateFnc,
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-button
            type="primary"
            :disabled="isStart || !form.is_specify_loc_turbines_initial"
            @click="onSelectOnly('dir_turbine_loc')"
          >
            选择
          </el-button>

          <el-tag
            v-if="form.dir_turbine_loc"
            :closable="!isStart"
            @close="handlePathCloseOnly('dir_turbine_loc')"
          >
            {{ form.dir_turbine_loc }}
          </el-tag>
        </el-form-item>
      </div>
    </el-form>
  </div>
</template>

<script>
import Mixins from '../mixins';
import { validateInt, validateFloat } from '../utils';

export default {
  name: 'PanelFirst',
  mixins: [Mixins],
  props: ['isStart', 'historyConfig'],
  data() {
    return {
      form: {
        parameters_turbine: [],
        num_turbines: '',
        dist_threshold: '',
        turbine_setting: '',
        is_specify_loc_turbines_initial: false,
        dir_turbine_loc: '',
      },
      rules: {
        // parameters_turbine: [
        //   { required: true, message: '请选择', trigger: ['blur', 'change'] },
        // ],
        num_turbines: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: validateInt,
            trigger: ['blur', 'change'],
          },
        ],
        dist_threshold: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: validateFloat,
            trigger: ['blur', 'change'],
          },
        ],
        turbine_setting: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
      },
    };
  },
  watch: {
    // eslint-disable-next-line func-names
    'form.is_specify_loc_turbines_initial': function (val) {
      if (!val) {
        this.form.dir_turbine_loc = '';
        this.$refs.form.clearValidate('dir_turbine_loc');
      }
    },
    historyConfig: {
      deep: true,
      handler(val) {
        this.$refs.form.resetFields();
        const keys = Object.keys(this.form);
        keys.forEach((key) => {
          this.form[key] = val[key] || this.form[key];
        });
      },
    },
  },
  methods: {
    validateFnc(rule, value, callback) {
      if (this.form.is_specify_loc_turbines_initial && !value) {
        callback('请选择');
      }
      callback();
    },
    validate() {
      return new Promise((resolve) => {
        this.$refs.form.validate((valid) => {
          const { is_specify_loc_turbines_initial } = this.form;
          resolve({
            valid,
            form: {
              ...this.form,
              is_specify_loc_turbines_initial:
                is_specify_loc_turbines_initial || '',
              parameters_turbine: undefined,
            },
          });
        });
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
}

.form-panel {
  flex: 1;
  display: flex;
  align-items: flex-start;

  .left,
  .right {
    width: 800px;
  }

  .tag-panel {
    width: 100%;
    max-height: 150px;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 6px 12px;
    border: 1px solid #ccc;
    margin-top: 8px;
    box-sizing: border-box;
  }

  .el-tag {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 6px 0;
  }

  .left {
    margin-right: 48px;
  }
}
</style>

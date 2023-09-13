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
        <el-form-item prop="dir_working">
          <el-tooltip
            slot="label"
            effect="dark"
            content="设置工作路径。"
            placement="top"
          >
            <span>
              选择工作目录
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('dir_working', 'openDirectory')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.dir_working"
            :closable="!isStart"
            @close="handlePathCloseOnly('dir_working')"
          >
            {{ form.dir_working }}
          </el-tag>
        </el-form-item>

        <el-form-item prop="num_opt_iteration">
          <el-tooltip
            slot="label"
            effect="dark"
            content="超过该数则停止迭代。由于可以随时中止，所以该值足够大即可。建议取值1000以上。"
            placement="top"
          >
            <span>
              优化迭代总数
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input
            :disabled="isStart"
            v-model="form.num_opt_iteration"
          ></el-input>
        </el-form-item>
        <el-form-item label="step_check" prop="step_check">
          <el-tooltip
            slot="label"
            effect="dark"
            content="每隔多个迭代步即自动保存，防止意外断电，可以断点续算。"
            placement="top"
          >
            <span>
              自动保存间隔
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input :disabled="isStart" v-model="form.step_check"></el-input>
        </el-form-item>
        <el-form-item label="name_to_save" prop="name_to_save">
          <el-tooltip
            slot="label"
            effect="dark"
            content="自动保存时的文件名。"
            placement="top"
          >
            <span>
              保存文件名
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input :disabled="isStart" v-model="form.name_to_save"></el-input>
        </el-form-item>
        <el-form-item label="name_to_load" prop="name_to_load">
          <el-tooltip
            slot="label"
            effect="dark"
            content="选择继续计算时，该项生效，即指定读取文件的文件名。"
            placement="top"
          >
            <span>
              读取文件名
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input :disabled="isStart" v-model="form.name_to_load"></el-input>
        </el-form-item>
      </div>
      <div class="right">
        <el-form-item label="高级选项" prop="advanced">
          <el-tooltip
            slot="label"
            effect="dark"
            content="优化器参数微调。"
            placement="top"
          >
            <span>
              高级选项
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-checkbox
            :disabled="isStart"
            v-model="form.advanced"
          ></el-checkbox>
        </el-form-item>

        <el-form-item label="is_set_new_vel" prop="is_set_new_vel">
          <el-tooltip
            slot="label"
            effect="dark"
            content="续算时重新初始化风机数据。"
            placement="top"
          >
            <span>
              新风机
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-checkbox
            :disabled="isStart || !form.advanced"
            v-model="form.is_set_new_vel"
          ></el-checkbox>
        </el-form-item>
        <el-form-item label="is_new_flow_field" prop="is_new_flow_field">
          <el-tooltip
            slot="label"
            effect="dark"
            content="续算时重新初始化风场数据。"
            placement="top"
          >
            <span>
              新风场
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-checkbox
            :disabled="isStart || !form.advanced"
            v-model="form.is_new_flow_field"
          ></el-checkbox>
        </el-form-item>

        <el-form-item
          label="seed_numpy"
          prop="seed_numpy"
          :rules="[
            {
              required: form.advanced,
              message: '请输入',
              trigger: ['blur', 'change'],
            },
            {
              validator: this.validateInt,
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-tooltip
            slot="label"
            effect="dark"
            content="限定随机性。"
            placement="top"
          >
            <span>
              随机种子
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input
            :disabled="isStart || !form.advanced"
            v-model="form.seed_numpy"
          ></el-input>
        </el-form-item>
        <el-form-item
          label="num_particles"
          prop="num_particles"
          :rules="[
            {
              required: form.advanced,
              message: '请输入',
              trigger: ['blur', 'change'],
            },
            {
              validator: this.validateInt,
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-tooltip
            slot="label"
            effect="dark"
            content="越多优化效果越好，也越慢。 建议取50~200。"
            placement="top"
          >
            <span>
              粒子群数目
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input
            :disabled="isStart || !form.advanced"
            v-model="form.num_particles"
          ></el-input>
        </el-form-item>
        <el-form-item
          label="fitness_initial"
          prop="fitness_initial"
          :rules="[
            {
              required: form.advanced,
              message: '请输入',
              trigger: ['blur', 'change'],
            },
            {
              validator: this.validateFloat,
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-tooltip
            slot="label"
            effect="dark"
            content="建议取0。"
            placement="top"
          >
            <span>
              粒子群初始值
              <i class="el-icon-question"></i>
            </span>
          </el-tooltip>
          <el-input
            :disabled="isStart || !form.advanced"
            v-model="form.fitness_initial"
          ></el-input>
        </el-form-item>
      </div>
    </el-form>

    <div class="btn-list">
      <el-button
        type="primary"
        :disabled="isStart"
        @click="onStart"
        class="save-btn"
      >
        新计算
      </el-button>
      <el-button
        type="primary"
        :disabled="isStart"
        @click="onContinue"
        class="save-btn"
      >
        继续计算
      </el-button>
      <el-button
        type="primary"
        :disabled="!isStart"
        @click="onStop"
        class="save-btn"
      >
        停止
      </el-button>
    </div>
  </div>
</template>

<script>
import Mixins from '../mixins';
import { validateFloat, validateInt } from '../utils';

export default {
  name: 'PanelFirst',
  mixins: [Mixins],
  props: ['isStart', 'historyConfig'],
  data() {
    return {
      form: {
        dir_working: '',
        num_opt_iteration: '',
        step_check: '',
        name_to_save: '',
        name_to_load: '',
        advanced: false,
        is_set_new_vel: false,
        is_new_flow_field: false,
        seed_numpy: 1,
        num_particles: 100,
        fitness_initial: 0,
      },
      rules: {
        dir_working: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
        num_opt_iteration: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: validateInt,
            trigger: ['blur', 'change'],
          },
        ],
        step_check: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: validateInt,
            trigger: ['blur', 'change'],
          },
        ],
        name_to_save: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
        ],
        name_to_load: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
        ],
      },
    };
  },
  watch: {
    // eslint-disable-next-line func-names
    'form.advanced': function (val) {
      if (!val) {
        this.form.is_set_new_vel = false;
        this.form.is_new_flow_field = false;
        this.form.seed_numpy = 1;
        this.form.num_particles = 100;
        this.form.fitness_initial = 0;
        this.$refs.form.clearValidate('seed_numpy');
        this.$refs.form.clearValidate('num_particles');
        this.$refs.form.clearValidate('fitness_initial');
      }
    },
    historyConfig: {
      deep: true,
      handler(val) {
        this.$refs.form.resetFields();
        const keys = Object.keys(this.form);
        const advancedKeys = ['is_set_new_vel', 'is_new_flow_field', 'seed_numpy', 'num_particles', 'fitness_initial'];
        let advanced = false;
        advancedKeys.forEach(k => {
          if (!advanced && val[k] !== this.form[k]) {
            advanced = true;
          }
        });
        this.form.advanced = advanced;

        keys.forEach((key) => {
          this.form[key] = val[key] || this.form[key];
        });
      },
    },
  },
  methods: {
    validateFloat,
    validateInt,
    validate() {
      return new Promise((resolve) => {
        this.$refs.form.validate((valid) => {
          const { is_set_new_vel, is_new_flow_field, fitness_initial } = this.form;
          resolve({
            valid,
            form: {
              ...this.form,
              advanced: undefined,
              fitness_initial: fitness_initial || '0.0',
              is_set_new_vel: is_set_new_vel || '',
              is_new_flow_field: is_new_flow_field || '',
            },
          });
        });
      });
    },

    onContinue() {
      this.$emit('start', true);
    },
    onStart() {
      this.$emit('start', false);
    },
    onStop() {
      this.$emit('stop', false);
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

.btn-list {
  display: flex;
  align-items: center;
  justify-content: center;
}

.save-btn {
  width: 120px;
  display: block;
}
</style>

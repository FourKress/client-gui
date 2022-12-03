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
        <el-form-item label="boundary_files" prop="boundary_files">
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelect('boundary_files')"
          >
            选择
          </el-button>
          <div class="tag-panel" v-if="form.boundary_files.length">
            <el-tag
              v-for="path in form.boundary_files"
              :key="path"
              :closable="!isStart"
              @close="handlePathClose(path, 'boundary_files')"
            >
              {{ path }}
            </el-tag>
          </div>
        </el-form-item>

        <el-form-item
          label="wind_files"
          prop="wind_file_item"
          :rules="[
            {
              required: !windList.length,
              message: '请选择',
              trigger: ['blur', 'change'],
            },
            {
              validator: validateFnc,
              trigger: ['blur', 'change'],
            },
          ]"
        >
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('wind_file_item')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.wind_file_item"
            :closable="!isStart"
            @close="handlePathCloseOnly('wind_file_item')"
          >
            {{ form.wind_file_item }}
          </el-tag>
        </el-form-item>

        <div class="form-row">
          <el-form-item
            style="width: 360px"
            label="wind_direction"
            prop="wind_direction_item"
            :rules="[
              {
                required: !windList.length,
                message: '请输入',
                trigger: ['blur', 'change'],
              },
              {
                validator: validateFloat,
                trigger: ['blur', 'change'],
              },
              {
                validator: validateFnc,
                trigger: ['blur', 'change'],
              },
            ]"
          >
            <el-input
              :disabled="isStart"
              v-model="form.wind_direction_item"
            ></el-input>
            <div class="tag-panel" style="width: 650px" v-if="windList.length">
              <el-tag
                v-for="(wind, index) in windList"
                :key="wind.path"
                :closable="!isStart"
                @close="handleWindPathClose(index)"
              >
                {{ wind.path }} / {{ wind.direction }}
              </el-tag>
            </div>
          </el-form-item>
          <div class="row-right">
            <el-button
              style="width: 60px; margin-left: 24px"
              type="primary"
              :disabled="isStart"
              @click="onAdd()"
            >
              添加
            </el-button>
          </div>
        </div>

        <el-form-item label="dir_ground_file" prop="dir_ground_file">
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('dir_ground_file')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.dir_ground_file"
            :closable="!isStart"
            @close="handlePathCloseOnly('dir_ground_file')"
          >
            {{ form.dir_ground_file }}
          </el-tag>
        </el-form-item>

        <el-form-item label="dir_mesh_file" prop="dir_mesh_file">
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('dir_mesh_file')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.dir_mesh_file"
            :closable="!isStart"
            @close="handlePathCloseOnly('dir_mesh_file')"
          >
            {{ form.dir_mesh_file }}
          </el-tag>
        </el-form-item>

        <el-form-item label="dir_measured_wind" prop="dir_measured_wind">
          <el-button
            type="primary"
            :disabled="isStart"
            @click="onSelectOnly('dir_measured_wind')"
          >
            选择
          </el-button>
          <el-tag
            v-if="form.dir_measured_wind"
            :closable="!isStart"
            @close="handlePathCloseOnly('dir_measured_wind')"
          >
            {{ form.dir_measured_wind }}
          </el-tag>
        </el-form-item>
      </div>
      <div class="right">
        <el-form-item label="height_mast" prop="height_mast">
          <el-input :disabled="isStart" v-model="form.height_mast"></el-input>
        </el-form-item>
        <el-form-item label="num_direction" prop="num_direction">
          <el-input :disabled="isStart" v-model="form.num_direction"></el-input>
        </el-form-item>
        <el-form-item label="num_speed" prop="num_speed">
          <el-input :disabled="isStart" v-model="form.num_speed"></el-input>
        </el-form-item>
        <el-form-item
          label="threshold_probability"
          prop="threshold_probability"
        >
          <el-input
            :disabled="isStart"
            v-model="form.threshold_probability"
          ></el-input>
        </el-form-item>
      </div>
    </el-form>
  </div>
</template>

<script>
import Mixins from '../mixins';
import { validateInt, validateFloat } from '../utils';

export default {
  name: 'PanelSecond',
  mixins: [Mixins],
  props: ['isStart', 'historyConfig'],
  data() {
    const checkValue = (rule, value, callback) => {
      if ((value && !/^\d+(?:\.\d+)?$/.test(value)) || value > 1 || value < 0) {
        callback('请输入0.0-1.0的整数或浮点数');
      }
      callback();
    };
    return {
      windList: [],
      form: {
        boundary_files: [],
        wind_files: [],
        wind_directions: [],
        wind_file_item: '',
        wind_direction_item: '',

        height_mast: '',
        num_direction: '',
        num_speed: '',
        threshold_probability: '',

        dir_ground_file: '',
        dir_mesh_file: '',
        dir_measured_wind: '',
      },
      rules: {
        boundary_files: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
        wind_files: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
        wind_directions: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
        ],

        height_mast: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
          {
            validator: validateFloat,
            trigger: ['blur', 'change'],
          },
        ],
        num_direction: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: validateInt,
            trigger: ['blur', 'change'],
          },
        ],
        num_speed: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
          {
            validator: validateInt,
            trigger: ['blur', 'change'],
          },
        ],
        threshold_probability: [
          { required: true, message: '请输入', trigger: ['blur', 'change'] },
          {
            validator: checkValue,
            trigger: ['blur', 'change'],
          },
        ],

        dir_ground_file: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
        dir_mesh_file: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
        dir_measured_wind: [
          { required: true, message: '请选择', trigger: ['blur', 'change'] },
        ],
      },
    };
  },
  watch: {
    historyConfig: {
      deep: true,
      handler(val) {
        this.$refs.form.resetFields();
        const keys = Object.keys(this.form);
        keys.forEach((key) => {
          if (['boundary_files', 'wind_files', 'wind_directions'].includes(key)) {
            this.form[key] = val[`list_${key}`];
          } else {
            this.form[key] = val[key];
          }
        });

        this.windList = this.form.wind_files.map((d, index) => ({
          direction: this.form.wind_directions[index],
          path: d,
        }));
      },
    },
  },
  mounted() {
  },
  methods: {
    validateFloat,

    onAdd() {
      this.$refs.form.validateField('wind_file_item');
      this.$refs.form.validateField('wind_direction_item');
      const { wind_direction_item, wind_file_item } = this.form;
      if (wind_direction_item && wind_file_item) {
        this.form.wind_directions.push(wind_direction_item);
        this.form.wind_files.push(wind_file_item);
        this.windList.push({
          direction: wind_direction_item,
          path: wind_file_item,
        });
        this.form.wind_direction_item = '';
        this.form.wind_file_item = '';
      }
    },

    handleWindPathClose(index) {
      this.windList.splice(index, 1);
      this.form.wind_directions.splice(index, 1);
      this.form.wind_files.splice(index, 1);
    },

    validateFnc(rule, value, callback) {
      if (!this.windList?.length) {
        if (rule.field === 'wind_file_item' && !this.form.wind_direction_item) {
          this.$refs.form.validateField('wind_direction_item');
        }
        if (rule.field === 'wind_direction_item') {
          if (!this.form.wind_file_item) {
            this.$refs.form.validateField('wind_file_item');
          } else {
            callback('请点击右侧添加按钮');
          }
        }
      }
      callback();
    },

    validate() {
      return new Promise((resolve) => {
        this.$refs.form.validate((valid) => {
          resolve({
            valid,
            form: this.form,
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

  .left {
    margin-right: 48px;
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

  .form-row {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
  }
}
</style>

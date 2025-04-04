// 等待DOM完全加载后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('热阻计算器已初始化');
    
    // 初始化函数
    function init() {
        // 这里将添加计算器的主要功能
    }

    init();
});

// 导入材料数据
import { getAllMaterials, getMaterialByName } from './data.js';

// 创建Vue应用
const app = Vue.createApp({
    data() {
        return {
            // 管道尺寸数据
            pipeInnerDiameter: '', // 管道内径d1
            pipeLength: '', // 管道长度l
            // 材料数据
            materials: getAllMaterials(),
            // 保温层数据（四层）
            layers: [
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 }
            ],
            // 计算结果
            thermalResistanceResults: [],
            totalCost: '0',
            costPerformance: '0',
            // 图表实例
            chart: null,
            // SVG实例
            svg: null,
            // 给定工况计算数据
            innerTemp: '',
            envTemp: '',
            heatTransferCoeff: '',
            temperatureResults: [],
            // 高亮行索引
            highlightIndex: -1,
            // 添加新的数据属性
            calculationResults: {
                totalResistance: 0,
                totalCost: 0,
                costPerformance: 0,
                layers: []
            },
            temperatureChart: null,
            materialImages: {
                '纳米气凝胶': './images/纳米气凝胶.jpg',
                '硅酸铝针刺毯': './images/硅酸铝针刺毯.jpg',
                '高温玻璃棉': './images/高温玻璃棉.jpg',
                '钠镁管保护层': './images/钠镁管保护层.jpg',
                '钠镁硬质保温块': './images/钠镁硬质保温块.jpg'
            },
            imageLoadError: false
        }
    },
    methods: {
        showHelp() {
            ElementPlus.ElMessage({
                message: '这里是帮助文档',
                type: 'info',
                duration: 3000
            });
        },
        refreshPage() {
            window.location.reload();
        },
        goBack() {
            // 暂时简单处理，后续可完善
            ElementPlus.ElMessage({
                message: '返回首页功能待实现',
                type: 'warning',
                duration: 3000
            });
        },
        // 获取材料标签
        getMaterialLabel(value) {
            const material = getMaterialByName(value);
            return material ? material.name : '';
        },
        // 处理材料选择变化
        handleMaterialChange() {
            this.calculateResults();
            this.updateDiagram();
        },
        // 计算每层的外径
        calculateDiameters() {
            let currentDiameter = parseFloat(this.pipeInnerDiameter) || 0;
            this.layers.forEach((layer, index) => {
                if (layer.thickness) {
                    currentDiameter += 2 * parseFloat(layer.thickness);
                    layer.diameter = currentDiameter;
                }
            });
        },
        
        // 计算材料用量
        calculateMaterialQuantity(layer, index) {
            const length = parseFloat(this.pipeLength) || 0;
            const innerDiameter = index === 0 ? 
                parseFloat(this.pipeInnerDiameter) : 
                this.layers[index - 1].diameter;
            const outerDiameter = layer.diameter;
            
            if (!innerDiameter || !outerDiameter || !length) return 0;
            
            const material = getMaterialByName(layer.material);
            if (!material) return 0;
            
            // 如果材料单位是"米"，直接返回长度
            if (material.unit === '米') {
                layer.quantity = length;
                return length;
            }
            
            // 否则计算体积（立方米）
            const volume = Math.PI * length * (Math.pow(outerDiameter, 2) - Math.pow(innerDiameter, 2)) / 4 / 1000000;
            layer.quantity = volume;
            return volume;
        },

        // 计算总成本
        calculateTotalCost() {
            this.calculateDiameters(); // 先计算所有层的外径
            
            let total = 0;
            this.layers.forEach((layer, index) => {
                if (layer.material && layer.thickness) {
                    const material = getMaterialByName(layer.material);
                    if (material) {
                        const quantity = this.calculateMaterialQuantity(layer, index);
                        const cost = quantity * material.price;
                        layer.cost = cost; // 保存每层的成本
                        total += cost;
                    }
                }
            });
            this.totalCost = total.toFixed(2);
        },

        // 计算热阻
        calculateThermalResistance() {
            this.calculateDiameters();
            this.calculationResults.layers = this.layers.map((layer, index) => {
                if (!layer.material || !layer.thickness) return null;
                
                const material = getMaterialByName(layer.material);
                if (!material) return null;
                
                const innerDiameter = index === 0 ? 
                    parseFloat(this.pipeInnerDiameter) : 
                    this.layers[index - 1].diameter;
                const outerDiameter = layer.diameter;
                const length = parseFloat(this.pipeLength) || 0;
                
                // 计算圆筒形保温层的热阻：R = ln(d2/d1)/(2πλl)
                const resistance = Math.log(outerDiameter / innerDiameter) / 
                    (2 * Math.PI * length * material.thermalConductivity);
                
                // 计算材料用量
                const quantity = this.calculateMaterialQuantity(layer, index);
                
                // 计算成本
                const cost = quantity * material.price;
                
                // 计算性价比：(K/W)/万元
                const costPerformance = (resistance / (cost / 10000)).toFixed(2);
                
                return {
                    layer: `第${index + 1}层`,
                    material: layer.material,
                    materialSpec: material.specification,
                    thermalConductivity: material.thermalConductivity,
                    thickness: layer.thickness,
                    innerDiameter: innerDiameter.toFixed(1),
                    outerDiameter: outerDiameter.toFixed(1),
                    quantity: quantity.toFixed(4),
                    unit: material.unit,
                    resistance: resistance.toFixed(4),
                    cost: cost.toFixed(2),
                    costPerformance: costPerformance
                };
            }).filter(result => result !== null);

            // 更新总计结果
            this.calculationResults.totalResistance = this.calculationResults.layers.reduce(
                (sum, item) => sum + parseFloat(item.resistance), 0
            ).toFixed(4);

            this.calculationResults.totalCost = this.calculationResults.layers.reduce(
                (sum, item) => sum + parseFloat(item.cost), 0
            ).toFixed(2);

            if (parseFloat(this.calculationResults.totalCost) > 0) {
                this.calculationResults.costPerformance = (
                    parseFloat(this.calculationResults.totalResistance) / 
                    (parseFloat(this.calculationResults.totalCost) / 10000)
                ).toFixed(2);
            }

            this.updateChart();
        },

        // 计算结果
        calculateResults() {
            if (!this.pipeInnerDiameter || !this.pipeLength) return;
            
            this.calculateThermalResistance();
            this.calculateTotalCost();
        },
        // 初始化图表
        initChart() {
            if (this.$refs.chartRef) {
                this.chart = echarts.init(this.$refs.chartRef);
                this.updateChart();
            }
        },
        // 更新图表
        updateChart() {
            if (!this.chart) return;

            const option = {
                title: {
                    text: '热阻、成本与性价比分析',
                    left: 'center'
                },
                tooltip: {
                    trigger: 'axis',
                    axisPointer: { type: 'shadow' },
                    formatter: function(params) {
                        let result = `${params[0].name}<br/>`;
                        params.forEach(param => {
                            let value = param.value;
                            // 根据不同的指标使用不同的小数位数
                            if (param.seriesName.includes('热阻')) {
                                value = value.toFixed(4);
                            } else if (param.seriesName.includes('成本')) {
                                value = value.toFixed(2);
                            } else if (param.seriesName.includes('性价比')) {
                                value = value.toFixed(2);
                            }
                            result += `${param.seriesName}: ${value}<br/>`;
                        });
                        return result;
                    }
                },
                legend: {
                    data: ['热阻(K/W)', '成本(元)', '性价比((K/W)/万元)'],
                    bottom: 0
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '15%',
                    containLabel: true
                },
                xAxis: {
                    type: 'category',
                    data: this.calculationResults.layers.map(item => item.layer)
                },
                yAxis: [
                    {
                        type: 'value',
                        name: '热阻(K/W)',
                        position: 'left',
                        axisLabel: {
                            formatter: '{value}'
                        },
                        splitLine: {
                            show: true,
                            lineStyle: {
                                type: 'dashed'
                            }
                        }
                    },
                    {
                        type: 'value',
                        name: '成本(元)',
                        position: 'right',
                        axisLabel: {
                            formatter: '{value}'
                        },
                        splitLine: {
                            show: false
                        }
                    }
                ],
                series: [
                    {
                        name: '热阻(K/W)',
                        type: 'bar',
                        data: this.calculationResults.layers.map(item => ({
                            value: parseFloat(item.resistance),
                            itemStyle: { color: '#91cc75' }
                        }))
                    },
                    {
                        name: '成本(元)',
                        type: 'bar',
                        yAxisIndex: 1,
                        data: this.calculationResults.layers.map(item => ({
                            value: parseFloat(item.cost),
                            itemStyle: { color: '#fac858' }
                        }))
                    },
                    {
                        name: '性价比((K/W)/万元)',
                        type: 'line',
                        smooth: true,
                        symbol: 'circle',
                        symbolSize: 8,
                        lineStyle: { width: 3 },
                        itemStyle: { color: '#ee6666' },
                        label: {
                            show: true,
                            position: 'top',
                            formatter: '{c}',
                            fontSize: 12,
                            color: '#ee6666'
                        },
                        data: this.calculationResults.layers.map(item => ({
                            value: parseFloat(item.costPerformance),
                            label: {
                                formatter: '{c}'
                            }
                        }))
                    }
                ]
            };

            this.chart.setOption(option, true);
        },
        // 初始化SVG示意图
        initDiagram() {
            const container = document.getElementById('svg-container');
            if (container) {
                this.svg = SVG().addTo(container).size('100%', '100%');
                this.svg.viewbox(0, 0, 400, 400);
                this.updateDiagram();
            }
        },
        // 更新示意图
        updateDiagram() {
            if (!this.svg) return;
            this.svg.clear();

            const centerX = 200;
            const centerY = 250;
            const scale = 160 / (parseFloat(this.pipeInnerDiameter) || 530);
            const baseRadius = (parseFloat(this.pipeInnerDiameter) || 530) * scale / 2;
            
            // 创建固定的提示框容器
            const tooltipContainer = document.createElement('div');
            tooltipContainer.style.position = 'absolute';
            tooltipContainer.style.display = 'none';
            tooltipContainer.style.backgroundColor = 'white';
            tooltipContainer.style.border = '1px solid #dcdfe6';
            tooltipContainer.style.borderRadius = '4px';
            tooltipContainer.style.padding = '8px 12px';
            tooltipContainer.style.boxShadow = '0 2px 12px rgba(0,0,0,0.1)';
            tooltipContainer.style.zIndex = '1000';
            tooltipContainer.style.fontSize = '14px';
            tooltipContainer.style.color = '#303133';
            tooltipContainer.style.pointerEvents = 'none';
            document.getElementById('svg-container').appendChild(tooltipContainer);
            
            const group = this.svg.group();
            let currentRadius = baseRadius;
            
            // 绘制管道本体（灰色）
            const pipe = group.circle(currentRadius * 2)
                .center(centerX, centerY)
                .fill('#A0A0A0')
                .stroke({ width: 1, color: '#666' });

            // 添加管道本体的鼠标事件
            pipe.mouseover(function(e) {
                const svgContainer = document.getElementById('svg-container');
                const rect = svgContainer.getBoundingClientRect();
                tooltipContainer.textContent = '管道本体层：管道本体';
                tooltipContainer.style.display = 'block';
                tooltipContainer.style.left = (e.clientX - rect.left + 10) + 'px';
                tooltipContainer.style.top = (e.clientY - rect.top - 30) + 'px';
            });
            
            pipe.mouseout(function() {
                tooltipContainer.style.display = 'none';
            });
            
            pipe.mousemove(function(e) {
                const svgContainer = document.getElementById('svg-container');
                const rect = svgContainer.getBoundingClientRect();
                tooltipContainer.style.left = (e.clientX - rect.left + 10) + 'px';
                tooltipContainer.style.top = (e.clientY - rect.top - 30) + 'px';
            });

            // 添加图例（横向两排布局）
            const legend = this.svg.group().translate(20, 20);
            
            // 根据实际填充的层数创建图例项
            const legendItems = [{ color: '#A0A0A0', text: '管道本体' }];
            
            // 只添加已填充材料的图例
            this.layers.forEach((layer, index) => {
                if (layer.material) {
                    legendItems.push({
                        color: this.getLayerColor(index),
                        text: layer.material
                    });
                }
            });

            // 每行显示3个图例项
            const itemsPerRow = 3;
            legendItems.forEach((item, i) => {
                const row = Math.floor(i / itemsPerRow);
                const col = i % itemsPerRow;
                const xOffset = col * 120;
                const yOffset = row * 25;

                legend.rect(15, 15)
                    .fill(item.color)
                    .stroke({ width: 1, color: '#666' })
                    .move(xOffset, yOffset);
                legend.text(item.text)
                    .font({ size: 12, family: 'Arial' })
                    .move(xOffset + 25, yOffset + 2);
            });
            
            // 存储所有环形的引用
            const rings = [];

            // 计算每层的温度分布
            for (let i = 0; i < 4; i++) {
                const layer = this.layers[i];
                if (layer.material && layer.thickness) {
                    const thickness = parseFloat(layer.thickness) * scale;
                    const nextRadius = currentRadius + thickness;
                    const material = getMaterialByName(layer.material);

                    // 创建环形
                    const ring = group.path(
                        `M ${centerX - nextRadius} ${centerY} ` +
                        `A ${nextRadius} ${nextRadius} 0 1 1 ${centerX + nextRadius} ${centerY} ` +
                        `A ${nextRadius} ${nextRadius} 0 1 1 ${centerX - nextRadius} ${centerY} ` +
                        `M ${centerX - currentRadius} ${centerY} ` +
                        `A ${currentRadius} ${currentRadius} 0 1 0 ${centerX + currentRadius} ${centerY} ` +
                        `A ${currentRadius} ${currentRadius} 0 1 0 ${centerX - currentRadius} ${centerY} Z`
                    )
                    .fill(this.getLayerColor(i))
                    .stroke({ width: 1, color: '#666' });

                    // 存储环形引用和相关信息
                    rings.push({
                        element: ring,
                        layer: layer,
                        material: material,
                        index: i
                    });

                    // 为环形添加点击事件
                    ring.click(() => {
                        this.showMaterialDetail(layer, i);
                    });

                    // 添加鼠标事件
                    ring.mouseover(function(e) {
                        const svgContainer = document.getElementById('svg-container');
                        const rect = svgContainer.getBoundingClientRect();
                        const layerText = i === 0 ? '最内层' : 
                                         i === 3 ? '最外层' : 
                                         `第${i + 1}层`;
                        tooltipContainer.textContent = `${layerText}：${layer.material}`;
                        tooltipContainer.style.display = 'block';
                        tooltipContainer.style.left = (e.clientX - rect.left + 10) + 'px';
                        tooltipContainer.style.top = (e.clientY - rect.top - 30) + 'px';
                    });
                    
                    ring.mouseout(function() {
                        tooltipContainer.style.display = 'none';
                    });
                    
                    ring.mousemove(function(e) {
                        const svgContainer = document.getElementById('svg-container');
                        const rect = svgContainer.getBoundingClientRect();
                        tooltipContainer.style.left = (e.clientX - rect.left + 10) + 'px';
                        tooltipContainer.style.top = (e.clientY - rect.top - 30) + 'px';
                    });

                    currentRadius = nextRadius;
                }
            }

            // 添加底部按钮组
            const buttonSpacing = 70;
            const buttonStartX = centerX - (buttonSpacing * 1.5);
            const buttonY = centerY + 180;
            
            // 创建四个按钮
            for (let i = 0; i < 4; i++) {
                const buttonGroup = this.svg.group().translate(buttonStartX + i * buttonSpacing, buttonY);
                
                const clickArea = buttonGroup.rect(60, 35)
                    .radius(8)
                    .fill('#409EFF')
                    .stroke({ width: 2, color: '#2c8cf4' })
                    .addClass('layer-button');
                
                const buttonText = i === 0 ? '最内层' : 
                                 i === 3 ? '最外层' : 
                                 `第${i + 1}层`;
                buttonGroup.text(buttonText)
                    .font({ size: 14, family: 'Arial', fill: '#fff', weight: 'bold' })
                    .center(30, 17.5);

                clickArea.mouseover(function() {
                    this.fill({ color: '#66b1ff' });
                    this.stroke({ width: 2, color: '#409EFF' });
                });
                
                clickArea.mouseout(function() {
                    this.fill({ color: '#409EFF' });
                    this.stroke({ width: 2, color: '#2c8cf4' });
                });

                clickArea.click(() => {
                    this.showMaterialDetail(this.layers[i], i);
                });
            }
        },
        
        // 获取层颜色
        getLayerColor(index) {
            const colors = [
                '#FFB6C1', // 浅粉红
                '#87CEEB', // 天蓝色
                '#98FB98', // 浅绿色
                '#DDA0DD'  // 梅红色
            ];
            return colors[index % colors.length];
        },
        
        // 计算温度分布
        calculateTemperatureDistribution(layerCount = 4, outerDiameter = null) {
            if (!this.innerTemp || !this.envTemp || !this.heatTransferCoeff) {
                ElementPlus.ElMessage.warning('请输入所有必要的参数');
                return;
            }
            
            // 获取基础参数
            const ts = parseFloat(this.innerTemp); // 蒸汽温度
            const t0 = parseFloat(this.envTemp);   // 环境温度
            const h = parseFloat(this.heatTransferCoeff); // 外对流传热系数
            
            // 使用传入的外径或获取最外层直径
            const dout = (outerDiameter || this.layers[layerCount - 1].diameter) / 1000;
            const pipeLength = parseFloat(this.pipeLength) || 0;
            const surfaceArea = Math.PI * dout * pipeLength;
            
            // 计算外部对流热阻
            const Rh = 1 / (h * surfaceArea);
            
            // 计算总热阻（外部对流热阻 + 保温层热阻之和）
            const insulationResistance = parseFloat(this.calculationResults.totalResistance) || 0;
            const Rsum = Rh + insulationResistance;
            
            // 计算散热功率
            const heatFlux = (ts - t0) / Rsum;
            
            // 初始化温度分布结果数组
            let temperatureResults = [];
            let currentTemp = ts;
            let accumulatedResistance = 0;
            
            // 计算每层的温度分布
            for (let i = 0; i < layerCount; i++) {
                const layerResistance = this.calculationResults.layers[i]?.resistance || 0;
                
                // 计算该层的内外温度和温差
                const t1 = currentTemp; // 内层温度
                const deltaT = heatFlux * layerResistance; // 温差
                const t2 = t1 - deltaT; // 外层温度
                const tAvg = (t1 + t2) / 2; // 平均温度
                
                temperatureResults.push({
                    layer: `第${i + 1}层`,
                    innerTemp: t1.toFixed(2),
                    outerTemp: t2.toFixed(2),
                    tempDiff: deltaT.toFixed(2),
                    avgTemp: tAvg.toFixed(2)
                });
                
                currentTemp = t2;
                accumulatedResistance += layerResistance;
            }
            
            // 保存计算结果
            this.temperatureResults = {
                layers: temperatureResults,
                pipeOuterDiameter: dout.toFixed(3),
                surfaceArea: surfaceArea.toFixed(2),
                externalResistance: Rh.toFixed(2),
                totalResistance: Rsum.toFixed(2),
                heatFlux: heatFlux.toFixed(2)
            };
            
            // 更新温度分布图表
            this.updateTemperatureChart();
        },
        
        // 更新温度分布图表
        updateTemperatureChart() {
            if (!this.temperatureChart) {
                this.initTemperatureChart();
            }

            if (!this.temperatureResults || !this.temperatureResults.layers) return;
            
            const option = {
                title: {
                    text: '保温层温度分布',
                    left: 'center',
                    top: 10,
                    textStyle: {
                        fontSize: 16,
                        color: '#303133'
                    }
                },
                tooltip: {
                    trigger: 'axis',
                    formatter: function(params) {
                        const data = params[0].data;
                        return `${params[0].name}<br/>
                                内层温度: ${data.innerTemp}°C<br/>
                                外层温度: ${data.outerTemp}°C<br/>
                                温差: ${data.tempDiff}°C<br/>
                                平均温度: ${data.avgTemp}°C`;
                    }
                },
                grid: {
                    top: 60,
                    bottom: 80,
                    left: 60,
                    right: 20  // 减小右侧边距
                },
                xAxis: {
                    type: 'category',
                    data: ['蒸汽', ...this.temperatureResults.layers.map(item => item.layer), '环境'],
                    axisLabel: {
                        interval: 0,
                        rotate: 0,
                        margin: 15,
                        padding: [5, 10, 5, 10]
                    },
                    axisTick: {
                        alignWithLabel: true
                    }
                },
                yAxis: {
                    type: 'value',
                    name: '温度(°C)',
                    nameLocation: 'middle',
                    nameGap: 40,
                    axisLabel: {
                        formatter: '{value}'
                    }
                },
                series: [
                    {
                        name: '温度分布',
                        type: 'line',
                        smooth: true,
                        symbolSize: 8,
                        lineStyle: {
                            width: 3,
                            color: '#409EFF'
                        },
                        itemStyle: {
                            color: '#409EFF'
                        },
                        areaStyle: {
                            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                                { offset: 0, color: 'rgba(64,158,255,0.3)' },
                                { offset: 1, color: 'rgba(64,158,255,0.1)' }
                            ])
                        },
                        data: [
                            { 
                                value: parseFloat(this.innerTemp),
                                innerTemp: this.innerTemp,
                                outerTemp: this.innerTemp,
                                tempDiff: 0,
                                avgTemp: this.innerTemp
                            },
                            ...this.temperatureResults.layers.map(item => ({
                                value: parseFloat(item.avgTemp),
                                innerTemp: item.innerTemp,
                                outerTemp: item.outerTemp,
                                tempDiff: item.tempDiff,
                                avgTemp: item.avgTemp
                            })),
                            {
                                value: parseFloat(this.envTemp),
                                innerTemp: this.envTemp,
                                outerTemp: this.envTemp,
                                tempDiff: 0,
                                avgTemp: this.envTemp
                            }
                        ],
                        label: {
                            show: true,
                            position: 'top',
                            formatter: '{@value}°C',
                            fontSize: 12,
                            color: '#606266'
                        }
                    }
                ]
            };
            
            this.temperatureChart.setOption(option, true);
        },

        // 初始化温度分布图表
        initTemperatureChart() {
            const chartDom = document.getElementById('temperatureChart');
            if (chartDom && !this.temperatureChart) {
                this.temperatureChart = echarts.init(chartDom);
                this.temperatureChart.setOption({
                    title: {
                        text: '保温层温度分布',
                        left: 'center',
                        top: 10,
                        textStyle: {
                            fontSize: 16,
                            color: '#303133'
                        }
                    },
                    grid: {
                        top: 60,
                        bottom: 80,
                        left: 60,
                        right: 20  // 减小右侧边距
                    },
                    xAxis: {
                        type: 'category',
                        data: ['暂无数据'],
                        axisLabel: {
                            interval: 0,
                            margin: 15,
                            padding: [5, 10, 5, 10]
                        }
                    },
                    yAxis: {
                        type: 'value',
                        name: '温度(°C)',
                        nameLocation: 'middle',
                        nameGap: 40
                    },
                    series: [{
                        type: 'line',
                        data: []
                    }]
                });
            }
        },
        // 表格行类名
        tableRowClassName({ row, rowIndex }) {
            if (rowIndex === this.highlightIndex) {
                return 'highlight-row';
            }
            return '';
        },
        calculateCondition() {
            // ... 现有的计算逻辑 ...
            
            // 在计算完成后更新图表
            this.updateTemperatureChart(results);
        },
        // 在窗口大小改变时调整图表大小
        handleResize() {
            if (this.temperatureChart) {
                this.temperatureChart.resize();
            }
        },
        // 处理图片加载错误
        handleImageError(event) {
            event.target.src = './images/default.jpg';  // 设置默认图片
            this.imageLoadError = true;
        },

        // 显示材料详情
        showMaterialDetail(layer, index) {
            if (layer.material) {
                const material = getMaterialByName(layer.material);
                const imagePath = this.materialImages[layer.material];
                
                ElementPlus.ElMessageBox.alert(
                    `<div class="material-dialog-content">
                        <img src="${imagePath}" 
                            alt="${layer.material}" 
                            onerror="this.onerror=null;this.src='./images/default.jpg';"
                            style="width: 100%; max-height: 300px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); margin-bottom: 20px;">
                        <div class="material-info">
                            <h3 style="color: #303133; margin-bottom: 15px;">${layer.material}</h3>
                            <p style="color: #606266; margin: 8px 0;">导热系数: ${material.thermalConductivity} W/(m·K)</p>
                            <p style="color: #606266; margin: 8px 0;">厚度: ${layer.thickness} mm</p>
                        </div>
                    </div>`,
                    `${index === 0 ? '最内' : index === 3 ? '最外' : '第' + (index + 1)}层材料详情`,
                    {
                        dangerouslyUseHTMLString: true,
                        confirmButtonText: '关闭',
                        customClass: 'material-dialog',
                        confirmButtonClass: 'el-button--primary',
                        closeOnClickModal: true,
                        closeOnPressEscape: true,
                        callback: (action) => {
                            if (this.imageLoadError) {
                                ElementPlus.ElMessage.warning('图片加载失败，请检查图片路径是否正确');
                                this.imageLoadError = false;
                            }
                        }
                    }
                );
            }
        },

        // 更新点击事件处理
        updateClickHandlers() {
            // 为环形添加点击事件
            const rings = document.querySelectorAll('.layer-ring');
            rings.forEach((ring, index) => {
                ring.addEventListener('click', () => {
                    const layer = this.layers[index];
                    this.showMaterialDetail(layer, index);
                });
            });

            // 为按钮添加点击事件
            const buttons = document.querySelectorAll('.layer-button');
            buttons.forEach((button, index) => {
                button.addEventListener('click', () => {
                    const layer = this.layers[index];
                    this.showMaterialDetail(layer, index);
                });
            });
        },

        // 应用基准方案
        applyScheme(scheme) {
            // 清空现有层
            this.layers = [
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 },
                { material: '', thickness: '', diameter: '', quantity: 0 }
            ];
            
            // 添加新层
            scheme.layers.forEach((layer, index) => {
                this.layers[index] = {
                    material: layer.material,
                    thickness: layer.thickness,
                    diameter: '',
                    quantity: 0
                };
            });
            
            // 设置其他参数
            this.pipeInnerDiameter = scheme.pipeInnerDiameter;
            this.pipeLength = scheme.pipeLength;
            this.innerTemp = scheme.steamTemperature;
            this.envTemp = scheme.ambientTemperature;
            this.heatTransferCoeff = scheme.externalConvectionCoefficient;
            
            // 重新计算
            this.calculateResults();
            
            // 根据方案层数计算温度分布
            if (scheme === baselineSchemes.scheme1) {
                // 方案一只有三层，使用第三层的外径
                const outerDiameter = this.layers[2].diameter || 0;
                this.calculateTemperatureDistribution(3, outerDiameter);
            } else {
                // 其他方案使用第四层的外径
                const outerDiameter = this.layers[3].diameter || 0;
                this.calculateTemperatureDistribution(4, outerDiameter);
            }
        },

        // 方案应用函数
        applyScheme1() {
            this.applyScheme(baselineSchemes.scheme1);
        },

        applyScheme2() {
            this.applyScheme(baselineSchemes.scheme2);
        },

        applyScheme3() {
            this.applyScheme(baselineSchemes.scheme3);
        },

        applyScheme4() {
            this.applyScheme(baselineSchemes.scheme4);
        }
    },
    watch: {
        // 监听所有可能影响示意图的数据变化
        pipeInnerDiameter: {
            handler(newVal) {
                this.calculateResults();
                this.updateDiagram();
            }
        },
        pipeLength: {
            handler(newVal) {
                this.calculateResults();
                this.updateDiagram();
            }
        },
        layers: {
            handler(newVal) {
                this.calculateResults();
                this.updateDiagram();
            },
            deep: true
        }
    },
    mounted() {
        // 监听窗口大小变化，调整图表大小
        window.addEventListener('resize', () => {
            if (this.chart) {
                this.chart.resize();
            }
        });
        
        // 初始化图表和示意图
        this.$nextTick(() => {
            this.initChart();
            this.initDiagram();
        });
        this.initTemperatureChart();
        this.updateClickHandlers();
    },
    created() {
        window.addEventListener('resize', this.handleResize);
    },
    beforeDestroy() {
        window.removeEventListener('resize', this.handleResize);
        if (this.temperatureChart) {
            this.temperatureChart.dispose();
        }
    }
});

// 注册Element Plus组件和图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component);
}

// 使用Element Plus
app.use(ElementPlus, {
    locale: ElementPlusLocaleZhCn,
});

// 挂载应用
app.mount('#app');

// 基准方案配置
const baselineSchemes = {
    scheme1: {
        layers: [
            { material: '纳米气凝胶', thickness: 20 },
            { material: '高温玻璃棉', thickness: 100 },
            { material: '钠镁管保护层', thickness: 2 }
        ],
        pipeInnerDiameter: 530,
        pipeLength: 2.5,
        steamTemperature: 271,
        ambientTemperature: 25,
        externalConvectionCoefficient: 2.96
    },
    scheme2: {
        layers: [
            { material: '纳米气凝胶', thickness: 10 },
            { material: '高温玻璃棉', thickness: 50 },
            { material: '钠镁管保护层', thickness: 70 },
            { material: '钠镁管保护层', thickness: 5 }
        ],
        pipeInnerDiameter: 530,
        pipeLength: 2.5,
        steamTemperature: 271,
        ambientTemperature: 25,
        externalConvectionCoefficient: 2.96
    },
    scheme3: {
        layers: [
            { material: '纳米气凝胶', thickness: 10 },
            { material: '硅酸铝针刺毯', thickness: 50 },
            { material: '钠镁管保护层', thickness: 70 },
            { material: '钠镁管保护层', thickness: 2 }
        ],
        pipeInnerDiameter: 530,
        pipeLength: 2.5,
        steamTemperature: 271,
        ambientTemperature: 25,
        externalConvectionCoefficient: 2.96
    },
    scheme4: {
        layers: [
            { material: '纳米气凝胶', thickness: 10 },
            { material: '硅酸铝针刺毯', thickness: 50 },
            { material: '硅酸铝针刺毯', thickness: 50 },
            { material: '钠镁管保护层', thickness: 2 }
        ],
        pipeInnerDiameter: 530,
        pipeLength: 2.5,
        steamTemperature: 271,
        ambientTemperature: 25,
        externalConvectionCoefficient: 2.96
    }
};

// 创建SVG示意图
function createDiagram() {
    const container = document.getElementById('svg-container');
    if (!container) return;

    // 清空容器
    container.innerHTML = '';
    
    // 创建SVG画布
    const draw = SVG().addTo('#svg-container').size('100%', 400);
    
    // 计算圆环的参数
    const centerX = draw.width() / 2;
    const centerY = 200;
    const baseRadius = 80;
    const radiusStep = 20;
    
    // 创建固定的提示框
    const tooltip = document.createElement('div');
    tooltip.style.position = 'absolute';
    tooltip.style.display = 'none';
    tooltip.style.backgroundColor = 'white';
    tooltip.style.border = '1px solid #ccc';
    tooltip.style.padding = '8px 12px';
    tooltip.style.borderRadius = '4px';
    tooltip.style.boxShadow = '0 2px 12px rgba(0,0,0,0.1)';
    tooltip.style.zIndex = '1000';
    tooltip.style.pointerEvents = 'none';
    container.appendChild(tooltip);

    // 定义颜色映射
    const colorMap = {
        '管道本体': '#808080',
        '纳米气凝胶': '#FFB6C1',
        '高温玻璃棉': '#87CEEB',
        '钠镁管保护层': '#90EE90',
        '最外层': '#DDA0DD'
    };

    // 创建圆环
    const rings = [];
    const materials = ['管道本体', '纳米气凝胶', '高温玻璃棉', '钠镁管保护层', '最外层'];
    
    materials.forEach((material, index) => {
        const radius = baseRadius + index * radiusStep;
        const ring = draw.circle(radius * 2)
            .move(centerX - radius, centerY - radius)
            .fill('none')
            .stroke({ color: colorMap[material], width: radiusStep - 2 });
            
        // 添加鼠标事件
        ring.on('mousemove', (e) => {
            const rect = container.getBoundingClientRect();
            const x = e.clientX - rect.left + 10;
            const y = e.clientY - rect.top - 30;
            
            tooltip.textContent = material;
            tooltip.style.display = 'block';
            tooltip.style.left = x + 'px';
            tooltip.style.top = y + 'px';
        });
        
        ring.on('mouseleave', () => {
            tooltip.style.display = 'none';
        });
        
        rings.push(ring);
    });
    
    // 创建图例
    const legendStartX = 20;
    const legendStartY = 350;
    const legendItemHeight = 25;
    const legendBoxWidth = 20;
    
    materials.forEach((material, index) => {
        // 图例色块
        draw.rect(legendBoxWidth, legendBoxWidth)
            .move(legendStartX, legendStartY + index * legendItemHeight)
            .fill(colorMap[material]);
        
        // 图例文字
        draw.text(material)
            .move(legendStartX + legendBoxWidth + 10, legendStartY + index * legendItemHeight)
            .font({
                family: 'Arial',
                size: 14,
                anchor: 'start'
            })
            .fill('#333');
    });
} 
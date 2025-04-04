// 材料参数数据
const MATERIAL_DATA = {
    // 纳米气凝胶
    nanoAerogel: {
        name: '纳米气凝胶',
        thermalConductivity: 0.02,      // 导热系数 W/(m·K)
        unit: '立方米',                 // 计价单位
        price: 13274.3438,             // 单价（元）
        specification: '10mm',          // 规格
        weight: 207                     // 容重
    },
    // 高温玻璃棉
    highTempGlassWool: {
        name: '高温玻璃棉',
        thermalConductivity: 0.04,      // 导热系数 W/(m·K)
        unit: '立方米',                 // 计价单位
        price: 796.4615,               // 单价（元）
        specification: '50mm',          // 规格
        weight: 48                      // 容重
    },
    // 硅酸铝针刺毯
    aluminosilicateNeedleMat: {
        name: '硅酸铝针刺毯',
        thermalConductivity: 0.09,      // 导热系数 W/(m·K)
        unit: '立方米',                 // 计价单位
        price: 867.25,                 // 单价（元）
        specification: '50mm',          // 规格
        weight: 128                     // 容重
    },
    // 钠镁硬质保温块
    steelMagnesiumBlock: {
        name: '钠镁硬质保温块',
        thermalConductivity: 0.05,      // 导热系数 W/(m·K)
        unit: '米',                     // 计价单位
        price: 256.638,                // 单价（元）
        specification: 'Φ790x70',       // 规格
        weight: 260                     // 容重
    },
    // 钠镁管保护层
    steelMagnesiumProtection: {
        name: '钠镁管保护层',
        thermalConductivity: 0.05,      // 导热系数 W/(m·K)
        unit: '米',                     // 计价单位
        price: 336.284,                // 单价（元）
        specification: 'Φ770x2/Φ790x2', // 规格
        weight: 260                     // 容重
    }
};

// 获取所有材料列表的函数
function getAllMaterials() {
    return Object.values(MATERIAL_DATA).map(material => ({
        value: material.name,
        label: material.name,
        ...material
    }));
}

// 根据材料名称获取材料数据的函数
function getMaterialByName(name) {
    return Object.values(MATERIAL_DATA).find(material => material.name === name);
}

// 导出数据和函数
export {
    MATERIAL_DATA,
    getAllMaterials,
    getMaterialByName
}; 
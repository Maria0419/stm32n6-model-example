/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : extmem_manager.c
  * @version        : 1.0.0
  * @brief          : This file implements the extmem configuration
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "extmem_manager.h"
#include "main.h"
#include <string.h>

/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* USER CODE BEGIN PV */
/* Private variables ---------------------------------------------------------*/

/* USER CODE END PV */

/* USER CODE BEGIN PFP */
/* Private function prototypes -----------------------------------------------*/

/* USER CODE END PFP */

/*
 * -- Insert your variables declaration here --
 */
/* USER CODE BEGIN 0 */
static void EXTMEM_ErrorHandler(void)
{
  Error_Handler();
}

/* USER CODE END 0 */

/*
 * -- Insert your external function declaration here --
 */
/* USER CODE BEGIN 1 */
void EXTMEM_XSPI2_MspInit(uint32_t *xspim_clk_refcount)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_XSPI2;
  PeriphClkInitStruct.Xspi2ClockSelection = RCC_XSPI2CLKSOURCE_IC3;
  PeriphClkInitStruct.ICSelection[RCC_IC3].ClockSelection = RCC_ICCLKSOURCE_PLL1;
  PeriphClkInitStruct.ICSelection[RCC_IC3].ClockDivider = 12;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    EXTMEM_ErrorHandler();
  }

  if ((*xspim_clk_refcount)++ == 0U)
  {
    __HAL_RCC_XSPIM_CLK_ENABLE();
  }
  __HAL_RCC_XSPI2_CLK_ENABLE();

  __HAL_RCC_GPION_CLK_ENABLE();
  GPIO_InitStruct.Pin = OCTOSPI_IO2_Pin | OCTOSPI_CLK_Pin | OCTOSPI_IO4_Pin | OCTOSPI_DQS_Pin |
                        OCTOSPI_IO1_Pin | OCTOSPI_IO3_Pin | OCTOSPI_NCS_Pin | OCTOSPI_IO5_Pin |
                        OCTOSPI_IO0_Pin | OCTOSPI_IO6_Pin | OCTOSPI_IO7_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF9_XSPIM_P2;
  HAL_GPIO_Init(GPION, &GPIO_InitStruct);
}

void EXTMEM_XSPI2_MspDeInit(uint32_t *xspim_clk_refcount)
{
  if (*xspim_clk_refcount > 0U)
  {
    (*xspim_clk_refcount)--;
  }
  if (*xspim_clk_refcount == 0U)
  {
    __HAL_RCC_XSPIM_CLK_DISABLE();
  }
  __HAL_RCC_XSPI2_CLK_DISABLE();

  HAL_GPIO_DeInit(GPION, OCTOSPI_IO2_Pin | OCTOSPI_CLK_Pin | OCTOSPI_IO4_Pin | OCTOSPI_DQS_Pin |
                         OCTOSPI_IO1_Pin | OCTOSPI_IO3_Pin | OCTOSPI_NCS_Pin | OCTOSPI_IO5_Pin |
                         OCTOSPI_IO0_Pin | OCTOSPI_IO6_Pin | OCTOSPI_IO7_Pin);
}

void MX_XSPI2_Boot_Init(void)
{
  XSPIM_CfgTypeDef sXspiManagerCfg = {0};

  hxspi2.Instance = XSPI2;
  hxspi2.Init.FifoThresholdByte = 4;
  hxspi2.Init.MemoryMode = HAL_XSPI_SINGLE_MEM;
  hxspi2.Init.MemoryType = HAL_XSPI_MEMTYPE_MACRONIX;
  hxspi2.Init.MemorySize = HAL_XSPI_SIZE_1GB;
  hxspi2.Init.ChipSelectHighTimeCycle = 1;
  hxspi2.Init.FreeRunningClock = HAL_XSPI_FREERUNCLK_DISABLE;
  hxspi2.Init.ClockMode = HAL_XSPI_CLOCK_MODE_0;
  hxspi2.Init.WrapSize = HAL_XSPI_WRAP_NOT_SUPPORTED;
  hxspi2.Init.ClockPrescaler = 0;
  hxspi2.Init.SampleShifting = HAL_XSPI_SAMPLE_SHIFT_NONE;
  hxspi2.Init.DelayHoldQuarterCycle = HAL_XSPI_DHQC_ENABLE;
  hxspi2.Init.ChipSelectBoundary = HAL_XSPI_BONDARYOF_NONE;
  hxspi2.Init.MaxTran = 0;
  hxspi2.Init.Refresh = 0;
  hxspi2.Init.MemorySelect = HAL_XSPI_CSSEL_NCS1;
  if (HAL_XSPI_Init(&hxspi2) != HAL_OK)
  {
    EXTMEM_ErrorHandler();
  }

  sXspiManagerCfg.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
  sXspiManagerCfg.IOPort = HAL_XSPIM_IOPORT_2;
  sXspiManagerCfg.Req2AckTime = 1;
  if (HAL_XSPIM_Config(&hxspi2, &sXspiManagerCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE) != HAL_OK)
  {
    EXTMEM_ErrorHandler();
  }
}

void FSBL_BootMemory_Init(void)
{
  MX_XSPI2_Boot_Init();
  MX_EXTMEM_MANAGER_Init();
}

/* USER CODE END 1 */

/**
  * Init External memory manager
  * @retval None
  */
void MX_EXTMEM_MANAGER_Init(void)
{

  /* USER CODE BEGIN MX_EXTMEM_Init_PreTreatment */

  /* USER CODE END MX_EXTMEM_Init_PreTreatment */

  /* Initialization of the memory parameters */
  memset(extmem_list_config, 0x0, sizeof(extmem_list_config));

  /* EXTMEMORY_1 */
  extmem_list_config[0].MemType = EXTMEM_NOR_SFDP;
  extmem_list_config[0].Handle = (void*)&hxspi2;
  extmem_list_config[0].ConfigType = EXTMEM_LINK_CONFIG_8LINES;

  EXTMEM_Init(EXTMEMORY_1, HAL_RCCEx_GetPeriphCLKFreq(RCC_PERIPHCLK_XSPI2));

  /* USER CODE BEGIN MX_EXTMEM_Init_PostTreatment */

  /* USER CODE END MX_EXTMEM_Init_PostTreatment */
}

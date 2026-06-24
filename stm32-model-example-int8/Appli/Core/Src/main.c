/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
#include "main.h"

/* Private includes ----------------------------------------------------------*/
#include "app_x-cube-ai.h"
/* USER CODE BEGIN Includes */
#include <string.h>
#include "npu_init.h"
#include "mcu_cache.h"
#include "ll_aton_osal.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define APPLI_LED_GREEN_PORT         GPIOO
#define APPLI_LED_GREEN_PIN          GPIO_PIN_1
#define APPLI_LED_GREEN_CLK_ENABLE() __HAL_RCC_GPIOO_CLK_ENABLE()

#define UART_PROTOCOL_VERSION       (1U)
#define UART_HEADER_SIZE            (16U)
#define UART_RX_TIMEOUT_MS          (5000U)
#define UART_TX_TIMEOUT_MS          (5000U)

#define UART_MSG_HELLO              (0x01U)
#define UART_MSG_IMAGE              (0x02U)
#define UART_MSG_HELLO_ACK          (0x81U)
#define UART_MSG_RESULT             (0x82U)
#define UART_MSG_ERROR              (0xFFU)

#define UART_STATUS_OK              (0U)
#define UART_STATUS_BAD_MAGIC       (1U)
#define UART_STATUS_BAD_VERSION     (2U)
#define UART_STATUS_BAD_MESSAGE     (3U)
#define UART_STATUS_BAD_LENGTH      (4U)
#define UART_STATUS_BAD_CRC         (5U)
#define UART_STATUS_AI_INIT         (6U)
#define UART_STATUS_AI_RUN          (7U)
#define UART_STATUS_UART            (8U)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CACHEAXI_HandleTypeDef hcacheaxi;

UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
STAI_NETWORK_CONTEXT_DECLARE(uart_network_context, STAI_NETWORK_CONTEXT_SIZE)

static const uint8_t protocol_magic[4] = {'S', 'S', 'E', 'G'};
static const float input_scales[] = STAI_NETWORK_IN_1_SCALES;
static const int16_t input_offsets[] = STAI_NETWORK_IN_1_OFFSETS;
static const float output_scales[] = STAI_NETWORK_OUT_1_SCALES;
static const int16_t output_offsets[] = STAI_NETWORK_OUT_1_OFFSETS;

static stai_ptr input_buffers[STAI_NETWORK_IN_NUM];
static stai_ptr output_buffers[STAI_NETWORK_OUT_NUM];
static stai_return_code network_status = STAI_ERROR_GENERIC;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_CACHEAXI_Init(void);
void MX_USART1_UART_Init(void);
static void SystemIsolation_Config(void);
/* USER CODE BEGIN PFP */
static void APPLI_SetStatusLed(void);
static void AI_UART_Init(void);
static void AI_UART_Process(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void APPLI_SetStatusLed(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  APPLI_LED_GREEN_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;

  GPIO_InitStruct.Pin = APPLI_LED_GREEN_PIN;
  HAL_GPIO_Init(APPLI_LED_GREEN_PORT, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LED2_Pin;
  HAL_GPIO_Init(LED2_GPIO_Port, &GPIO_InitStruct);

  HAL_GPIO_WritePin(LED2_GPIO_Port, LED2_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(APPLI_LED_GREEN_PORT, APPLI_LED_GREEN_PIN, GPIO_PIN_SET);
}

static uint16_t read_u16_le(const uint8_t *data)
{
  return (uint16_t)data[0] | ((uint16_t)data[1] << 8);
}

static uint32_t read_u32_le(const uint8_t *data)
{
  return (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
         ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
}

static void write_u16_le(uint8_t *data, uint16_t value)
{
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
}

static void write_u32_le(uint8_t *data, uint32_t value)
{
  data[0] = (uint8_t)value;
  data[1] = (uint8_t)(value >> 8);
  data[2] = (uint8_t)(value >> 16);
  data[3] = (uint8_t)(value >> 24);
}

static void write_u64_le(uint8_t *data, uint64_t value)
{
  write_u32_le(data, (uint32_t)value);
  write_u32_le(data + 4, (uint32_t)(value >> 32));
}

static uint32_t crc32_update(uint32_t crc, const uint8_t *data, uint32_t length)
{
  for (uint32_t index = 0; index < length; index++)
  {
    crc ^= data[index];
    for (uint32_t bit = 0; bit < 8U; bit++)
    {
      crc = (crc >> 1) ^ (0xEDB88320U & (0U - (crc & 1U)));
    }
  }
  return crc;
}

static uint32_t crc32(const uint8_t *data, uint32_t length)
{
  return ~crc32_update(0xFFFFFFFFU, data, length);
}

static HAL_StatusTypeDef uart_send_frame(uint8_t message, uint16_t status,
                                         const uint8_t *payload, uint32_t payload_length)
{
  uint8_t header[UART_HEADER_SIZE];

  memcpy(header, protocol_magic, sizeof(protocol_magic));
  header[4] = UART_PROTOCOL_VERSION;
  header[5] = message;
  write_u16_le(header + 6, status);
  write_u32_le(header + 8, payload_length);
  write_u32_le(header + 12, payload_length == 0U ? 0U : crc32(payload, payload_length));

  if (HAL_UART_Transmit(&huart1, header, sizeof(header), UART_TX_TIMEOUT_MS) != HAL_OK)
  {
    return HAL_ERROR;
  }
  if (payload_length != 0U &&
      HAL_UART_Transmit(&huart1, payload, payload_length, UART_TX_TIMEOUT_MS) != HAL_OK)
  {
    return HAL_ERROR;
  }
  return HAL_OK;
}

static void uart_send_error(uint16_t status)
{
  (void)uart_send_frame(UART_MSG_ERROR, status, NULL, 0U);
}

static HAL_StatusTypeDef uart_send_result(const uint8_t *timing, const uint8_t *output)
{
  uint8_t header[UART_HEADER_SIZE];
  uint32_t payload_length = 8U + STAI_NETWORK_OUT_1_SIZE_BYTES;
  uint32_t checksum = crc32_update(0xFFFFFFFFU, timing, 8U);

  checksum = ~crc32_update(checksum, output, STAI_NETWORK_OUT_1_SIZE_BYTES);
  memcpy(header, protocol_magic, sizeof(protocol_magic));
  header[4] = UART_PROTOCOL_VERSION;
  header[5] = UART_MSG_RESULT;
  write_u16_le(header + 6, UART_STATUS_OK);
  write_u32_le(header + 8, payload_length);
  write_u32_le(header + 12, checksum);

  if (HAL_UART_Transmit(&huart1, header, sizeof(header), UART_TX_TIMEOUT_MS) != HAL_OK ||
      HAL_UART_Transmit(&huart1, timing, 8U, UART_TX_TIMEOUT_MS) != HAL_OK ||
      HAL_UART_Transmit(&huart1, output, STAI_NETWORK_OUT_1_SIZE_BYTES,
                        UART_TX_TIMEOUT_MS) != HAL_OK)
  {
    return HAL_ERROR;
  }
  return HAL_OK;
}

static void handle_hello(uint32_t payload_length)
{
  uint8_t payload[24];

  if (payload_length != 0U)
  {
    uart_send_error(UART_STATUS_BAD_LENGTH);
    return;
  }
  if (network_status != STAI_SUCCESS)
  {
    uart_send_error(UART_STATUS_AI_INIT);
    return;
  }

  write_u16_le(payload, STAI_NETWORK_IN_1_WIDTH);
  write_u16_le(payload + 2, STAI_NETWORK_IN_1_HEIGHT);
  write_u32_le(payload + 4, STAI_NETWORK_IN_1_SIZE_BYTES);
  write_u32_le(payload + 8, STAI_NETWORK_OUT_1_SIZE_BYTES);
  memcpy(payload + 12, &input_scales[0], sizeof(float));
  write_u16_le(payload + 16, (uint16_t)input_offsets[0]);
  memcpy(payload + 18, &output_scales[0], sizeof(float));
  write_u16_le(payload + 22, (uint16_t)output_offsets[0]);
  (void)uart_send_frame(UART_MSG_HELLO_ACK, UART_STATUS_OK, payload, sizeof(payload));
}

static stai_return_code run_network(uint64_t *inference_us)
{
  stai_return_code result;
  stai_return_code execution_status;
  uint32_t elapsed_cycles;

  mcu_cache_clean_invalidate_range((uint32_t)input_buffers[0],
                                   (uint32_t)input_buffers[0] + STAI_NETWORK_IN_1_SIZE_BYTES);
  mcu_cache_invalidate_range((uint32_t)output_buffers[0],
                             (uint32_t)output_buffers[0] + STAI_NETWORK_OUT_1_SIZE_BYTES);

  execution_status = stai_ext_network_get_nn_run_status(uart_network_context);
  if (execution_status == STAI_DONE || execution_status == STAI_RUNNING_NO_WFE)
  {
    result = stai_ext_network_new_inference(uart_network_context);
    if (result != STAI_SUCCESS)
    {
      return result;
    }
  }
  else if (execution_status != STAI_SUCCESS)
  {
    return execution_status;
  }

  DWT->CYCCNT = 0U;
  result = stai_network_run(uart_network_context, STAI_MODE_ASYNC);
  while (result == STAI_RUNNING_WFE || result == STAI_RUNNING_NO_WFE)
  {
    if (result == STAI_RUNNING_WFE)
    {
      LL_ATON_OSAL_WFE();
    }
    result = stai_ext_network_run_continue(uart_network_context);
  }
  elapsed_cycles = DWT->CYCCNT;

  *inference_us = ((uint64_t)elapsed_cycles * 1000000ULL) / SystemCoreClock;
  return result;
}

static void handle_image(uint32_t payload_length, uint32_t expected_crc)
{
  uint8_t timing[8];
  uint64_t inference_us;
  stai_return_code result;

  if (network_status != STAI_SUCCESS)
  {
    uart_send_error(UART_STATUS_AI_INIT);
    return;
  }
  if (payload_length != STAI_NETWORK_IN_1_SIZE_BYTES)
  {
    uart_send_error(UART_STATUS_BAD_LENGTH);
    return;
  }
  if (HAL_UART_Receive(&huart1, input_buffers[0], payload_length, UART_RX_TIMEOUT_MS) != HAL_OK)
  {
    uart_send_error(UART_STATUS_UART);
    return;
  }
  if (crc32(input_buffers[0], payload_length) != expected_crc)
  {
    uart_send_error(UART_STATUS_BAD_CRC);
    return;
  }

  result = run_network(&inference_us);
  if (result != STAI_DONE && result != STAI_SUCCESS)
  {
    uart_send_error(UART_STATUS_AI_RUN);
    return;
  }

  write_u64_le(timing, inference_us);
  (void)uart_send_result(timing, output_buffers[0]);
}

static void AI_UART_Init(void)
{
  stai_size input_count = 0U;
  stai_size output_count = 0U;

  aiPreInitialize();
  mcu_cache_clean_invalidate();
  network_status = stai_runtime_init();
  if (network_status == STAI_SUCCESS)
  {
    network_status = user_stai_network_init(uart_network_context);
  }
  if (network_status == STAI_SUCCESS)
  {
    network_status = stai_network_get_inputs(uart_network_context, input_buffers, &input_count);
  }
  if (network_status == STAI_SUCCESS)
  {
    network_status = stai_network_get_outputs(uart_network_context, output_buffers, &output_count);
  }
  if (network_status == STAI_SUCCESS &&
      (input_count != STAI_NETWORK_IN_NUM || output_count != STAI_NETWORK_OUT_NUM))
  {
    network_status = STAI_ERROR_GENERIC;
  }

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void AI_UART_Process(void)
{
  uint8_t header[UART_HEADER_SIZE];
  uint32_t payload_length;
  uint32_t payload_crc;

  if (HAL_UART_Receive(&huart1, header, sizeof(header), HAL_MAX_DELAY) != HAL_OK)
  {
    uart_send_error(UART_STATUS_UART);
    return;
  }
  if (memcmp(header, protocol_magic, sizeof(protocol_magic)) != 0)
  {
    uart_send_error(UART_STATUS_BAD_MAGIC);
    return;
  }
  if (header[4] != UART_PROTOCOL_VERSION)
  {
    uart_send_error(UART_STATUS_BAD_VERSION);
    return;
  }
  if (read_u16_le(header + 6) != UART_STATUS_OK)
  {
    uart_send_error(UART_STATUS_BAD_MESSAGE);
    return;
  }

  payload_length = read_u32_le(header + 8);
  payload_crc = read_u32_le(header + 12);
  if (header[5] == UART_MSG_HELLO)
  {
    handle_hello(payload_length);
  }
  else if (header[5] == UART_MSG_IMAGE)
  {
    handle_image(payload_length, payload_crc);
  }
  else
  {
    uart_send_error(UART_STATUS_BAD_MESSAGE);
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  APPLI_SetStatusLed();
  MX_CACHEAXI_Init();
  
  SystemIsolation_Config();
  /* USER CODE BEGIN 2 */
  MX_USART1_UART_Init();
  AI_UART_Init();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    AI_UART_Process();
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief CACHEAXI Initialization Function
  * @param None
  * @retval None
  */
static void MX_CACHEAXI_Init(void)
{

  /* USER CODE BEGIN CACHEAXI_Init 0 */

  /* USER CODE END CACHEAXI_Init 0 */

  /* USER CODE BEGIN CACHEAXI_Init 1 */

  /* USER CODE END CACHEAXI_Init 1 */
  hcacheaxi.Instance = CACHEAXI;
  if (HAL_CACHEAXI_Init(&hcacheaxi) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CACHEAXI_Init 2 */

  /* USER CODE END CACHEAXI_Init 2 */

}

/**
  * @brief RIF Initialization Function
  * @param None
  * @retval None
  */
  static void SystemIsolation_Config(void)
{

  /* USER CODE BEGIN RIF_Init 0 */

  /* USER CODE END RIF_Init 0 */

  /* set all required IPs as secure privileged */
  __HAL_RCC_RIFSC_CLK_ENABLE();

  /*RIMC configuration*/
  RIMC_MasterConfig_t RIMC_master = {0};
  RIMC_master.MasterCID = RIF_CID_1;
  RIMC_master.SecPriv = RIF_ATTRIBUTE_SEC | RIF_ATTRIBUTE_NPRIV;
  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_ETH1, &RIMC_master);

  HAL_RIF_RIMC_ConfigMasterAttributes(RIF_MASTER_INDEX_SDMMC2, &RIMC_master);

  /* RIF-Aware IPs Config */

  /* set up GPIO configuration */
  HAL_GPIO_ConfigPinAttributes(GPIOA,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOB,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOC,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOD,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOE,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOF,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOG,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOH,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPION,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOO,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_8,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_9,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_10,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_11,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_12,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_13,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_14,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOP,GPIO_PIN_15,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_0,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_1,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_2,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_3,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_4,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_5,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_6,GPIO_PIN_SEC|GPIO_PIN_NPRIV);
  HAL_GPIO_ConfigPinAttributes(GPIOQ,GPIO_PIN_7,GPIO_PIN_SEC|GPIO_PIN_NPRIV);

  /* USER CODE BEGIN RIF_Init 1 */

  /* USER CODE END RIF_Init 1 */
  /* USER CODE BEGIN RIF_Init 2 */

  /* USER CODE END RIF_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOQ_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOO_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPION_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOQ, LCD_BL_CTRL_Pin|GPIO_PIN_3|PWR_SD_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(EN_MODULE_GPIO_Port, EN_MODULE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, PWR_USB2_EN_Pin|AUDIO_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LCD_NRST_GPIO_Port, LCD_NRST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SD_SEL_GPIO_Port, SD_SEL_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB1_OCP_GPIO_Port, USB1_OCP_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : LCD_BL_CTRL_Pin PQ3 PWR_SD_EN_Pin */
  GPIO_InitStruct.Pin = LCD_BL_CTRL_Pin|GPIO_PIN_3|PWR_SD_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOQ, &GPIO_InitStruct);

  /*Configure GPIO pin : USB1_INT_Pin */
  GPIO_InitStruct.Pin = USB1_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB1_INT_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : EN_MODULE_Pin */
  GPIO_InitStruct.Pin = EN_MODULE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(EN_MODULE_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PQ4 TOF_LPn_Pin IMU_INT2_Pin IMU_INT1_Pin
                           TOF_INT_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_4|TOF_LPn_Pin|IMU_INT2_Pin|IMU_INT1_Pin
                          |TOF_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOQ, &GPIO_InitStruct);

  /*Configure GPIO pin : NRST_CAM_Pin */
  GPIO_InitStruct.Pin = NRST_CAM_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(NRST_CAM_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : PWR_USB2_EN_Pin AUDIO_RST_Pin */
  GPIO_InitStruct.Pin = PWR_USB2_EN_Pin|AUDIO_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : LCD_NRST_Pin */
  GPIO_InitStruct.Pin = LCD_NRST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LCD_NRST_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : SD_SEL_Pin */
  GPIO_InitStruct.Pin = SD_SEL_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(SD_SEL_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LED2_Pin */
  GPIO_InitStruct.Pin = LED2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(LED2_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB1_OCP_Pin */
  GPIO_InitStruct.Pin = USB1_OCP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB1_OCP_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

Feature: Test Feature for SXM-2302295
  
  Scenario: Test Scenario for Timeline Generation
    Given the IP channel is created
    And the application is tuned to channel 512
    When the application is displaying the linear tuner screen with the channel 561 tile in focus
    Then the track name shall meet the following requirements
    
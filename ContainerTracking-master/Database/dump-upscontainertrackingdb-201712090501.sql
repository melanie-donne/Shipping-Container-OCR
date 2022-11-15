-- MySQL dump 10.16  Distrib 10.2.11-MariaDB, for Win64 (AMD64)
--
-- Host: localhost    Database: upscontainertrackingdb
-- ------------------------------------------------------
-- Server version	10.2.11-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `camera`
--

DROP TABLE IF EXISTS `camera`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `camera` (
  `CameraID` bigint(20) NOT NULL AUTO_INCREMENT,
  `MAC_Address` varchar(100) COLLATE utf8_bin NOT NULL,
  `locationid` bigint(20) DEFAULT NULL,
  `DateAdded` datetime NOT NULL,
  PRIMARY KEY (`CameraID`),
  KEY `camera_location_fk` (`locationid`),
  CONSTRAINT `camera_location_fk` FOREIGN KEY (`locationid`) REFERENCES `location` (`locationid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Contains the CameraID and the MAC_Address of the camera';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `camera`
--

LOCK TABLES `camera` WRITE;
/*!40000 ALTER TABLE `camera` DISABLE KEYS */;
/*!40000 ALTER TABLE `camera` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `container`
--

DROP TABLE IF EXISTS `container`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `container` (
  `ContainerID` bigint(20) NOT NULL AUTO_INCREMENT,
  `LabelText` varchar(100) COLLATE utf8_bin NOT NULL,
  PRIMARY KEY (`ContainerID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Contains the actual label text of the container';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `container`
--

LOCK TABLES `container` WRITE;
/*!40000 ALTER TABLE `container` DISABLE KEYS */;
/*!40000 ALTER TABLE `container` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `labelinfo`
--

DROP TABLE IF EXISTS `labelinfo`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `labelinfo` (
  `labelinfoid` bigint(20) NOT NULL AUTO_INCREMENT,
  `ContainerID` bigint(20) NOT NULL,
  `CameraID` bigint(20) NOT NULL,
  `ObjectJSON` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,
  `CaptureDate` datetime NOT NULL,
  `IsPriority` bit(1) NOT NULL,
  PRIMARY KEY (`labelinfoid`),
  KEY `labelinfo_container_fk` (`ContainerID`),
  KEY `labelinfo_camera_fk` (`CameraID`),
  CONSTRAINT `labelinfo_camera_fk` FOREIGN KEY (`CameraID`) REFERENCES `camera` (`CameraID`),
  CONSTRAINT `labelinfo_container_fk` FOREIGN KEY (`ContainerID`) REFERENCES `container` (`ContainerID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Contains the information about the label that was captured by the camera including the ContainerID, CameraID, JSON,Capture Date and Priority Status';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `labelinfo`
--

LOCK TABLES `labelinfo` WRITE;
/*!40000 ALTER TABLE `labelinfo` DISABLE KEYS */;
/*!40000 ALTER TABLE `labelinfo` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `location`
--

DROP TABLE IF EXISTS `location`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `location` (
  `locationid` bigint(20) NOT NULL AUTO_INCREMENT,
  `Building` varchar(100) COLLATE utf8_bin DEFAULT NULL,
  `PoleLocation` varchar(100) COLLATE utf8_bin DEFAULT NULL,
  PRIMARY KEY (`locationid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin COMMENT='Contains the physical location in the building';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `location`
--

LOCK TABLES `location` WRITE;
/*!40000 ALTER TABLE `location` DISABLE KEYS */;
/*!40000 ALTER TABLE `location` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping routines for database 'upscontainertrackingdb'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2017-12-09  5:01:39

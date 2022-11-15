SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

USE [master];
GO

IF EXISTS (SELECT * FROM sys.databases WHERE name = 'UPSCapstone')
	DROP DATABASE UPSCapstone;
GO

-- Create the UPSCapstone database.
CREATE DATABASE UPSCapstone;
GO

-- Specify a simple recovery model to keep the log growth to a minimum.
ALTER DATABASE UPSCapstone 
	SET RECOVERY SIMPLE;
GO

USE UPSCapstone;
GO

-- Create the ULD_LabelInfo table.
IF NOT EXISTS (SELECT * FROM sys.objects 
		WHERE object_id = OBJECT_ID(N'[dbo].[LabelInfo]') 
		AND type in (N'U'))
BEGIN
CREATE TABLE [dbo].[LabelInfo](
	[LabelInfoID] [int] IDENTITY(1,1) NOT NULL,
	[ContainerID] [bigint] NOT NULL,
	[CameraID] [int] NOT NULL,
	[ObjectJSON] [nvarchar](MAX) NULL,
	[CaptureDate] [datetime] NOT NULL,
	[IsPriority] [bit] NOT NULL,
 CONSTRAINT [PK_LabelID] PRIMARY KEY CLUSTERED 
(
	[LabelInfoID] ASC
)WITH (IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
END
GO

-- Create the Container table.
IF NOT EXISTS (SELECT * FROM sys.objects 
		WHERE object_id = OBJECT_ID(N'[dbo].[Container]') 
		AND type in (N'U'))
BEGIN
CREATE TABLE [dbo].[Container](
	[ContainerID] [bigint] IDENTITY(1,1) NOT NULL,
	[LableText] [nvarchar](20) NULL,
	[IsPriority] [bit] NOT NULL,
	--[TypeCode] [nvarchar](3) NOT NULL,
	--[SerialNumber] [int] NOT NULL,
	--[OwnerCode] [nvarchar](4) NOT NULL,
	--[ExtendedCode] [nvarchar](3) NULL,
 CONSTRAINT [PK_ContainerID] PRIMARY KEY CLUSTERED 
(
	[ContainerID] ASC
)WITH (IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
END
GO


-- Create the Camera table.
IF NOT EXISTS (SELECT * FROM sys.objects 
		WHERE object_id = OBJECT_ID(N'[dbo].[Camera]') 
		AND type in (N'U'))
BEGIN
CREATE TABLE [dbo].[Camera](
	[CameraID] [int] IDENTITY(1,1) NOT NULL,
	[MAC_Address] [nvarchar](100) NOT NULL,
	[LocationID] [int] NULL,
	[DateAdded] [datetime] NOT NULL,
 CONSTRAINT [PK_CameraID] PRIMARY KEY CLUSTERED 
(
	[CameraID] ASC
)WITH (IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
END
GO


-- Create the Location table.
IF NOT EXISTS (SELECT * FROM sys.objects 
		WHERE object_id = OBJECT_ID(N'[dbo].[Location]') 
		AND type in (N'U'))
BEGIN
CREATE TABLE [dbo].[Location](
	[LocationID] [int] IDENTITY(1,1) NOT NULL,
	[Building] [nvarchar](50) NULL,
	[PoleLocation] [nvarchar](20) NULL,
 CONSTRAINT [PK_LocationID] PRIMARY KEY CLUSTERED 
(
	[LocationID] ASC
)WITH (IGNORE_DUP_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
END
GO


ALTER TABLE [dbo].[LabelInfo]  WITH CHECK 
ADD  CONSTRAINT [FK_LabelInfo_Container] FOREIGN KEY([ContainerID])
REFERENCES [dbo].[Container] ([ContainerID])
GO

ALTER TABLE [dbo].[LabelInfo] CHECK CONSTRAINT [FK_LabelInfo_Container]
GO

ALTER TABLE [dbo].[LabelInfo]  WITH CHECK 
ADD  CONSTRAINT [FK_LabelInfo_Camera] FOREIGN KEY([CameraID])
REFERENCES [dbo].[Camera] ([CameraID])
GO

ALTER TABLE [dbo].[LabelInfo] CHECK CONSTRAINT [FK_LabelInfo_Camera]
GO


ALTER TABLE [dbo].[Camera]  WITH CHECK 
ADD  CONSTRAINT [FK_Camera_Location] FOREIGN KEY([LocationID])
REFERENCES [dbo].[Location] ([LocationID])
GO

ALTER TABLE [dbo].[Camera] CHECK CONSTRAINT [FK_Camera_Location]
GO

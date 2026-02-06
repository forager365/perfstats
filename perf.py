 pip install boto3 sparklines

 rds_io_analyzer.py
python rds_io_analyzer.py --profile my-sso-profile --region us-east-1 --db-identifier my-production-db
 import boto3
import argparse
import sys
from datetime import datetime, timedelta
from statistics import mean
import sparklines  # pip install sparklines

def get_args():
    parser = argparse.ArgumentParser(description="RDS/Aurora Storage I/O Rightsizing Analyzer")
    parser.add_argument("--profile", required=True, help="AWS SSO Profile Name")
    parser.add_argument("--region", required=True, help="AWS Region")
    parser.add_argument("--db-identifier", required=True, help="RDS Instance or Aurora Cluster Identifier")
    return parser.parse_args()

def get_cw_metric(cw_client, namespace, metric, db_id, start, end):
    """Fetches Max and Avg metrics for the last 15 days with 1-hour granularity"""
    try:
        response = cw_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_id}],
            StartTime=start,
            EndTime=end,
            Period=3600, # 1 Hour granularity for 15 day overview
            Statistics=['Average', 'Maximum']
        )
        # Sort by timestamp to ensure sparkline order
        return sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
    except Exception as e:
        print(f"Error fetching {metric}: {e}")
        return []

def print_spark(label, data_points, key='Maximum', unit='Count'):
    if not data_points:
        print(f"{label}: No Data")
        return 0, 0

    values = [dp[key] for dp in data_points]
    if not values:
        return 0, 0
        
    avg_val = mean(values)
    max_val = max(values)
    
    # Create sparkline
    spark = sparklines.sparklines(values)[0]
    
    # Format string
    fmt_str = f"{label:<20} |{spark}| Max: {max_val:.2f} {unit}, Avg: {avg_val:.2f} {unit}"
    print(fmt_str)
    return max_val, avg_val

def assess_io(db_info, read_iops, write_iops, read_thru, write_thru, queue_depth):
    print("\n--- 🧠 Rightsizing Assessment ---")
    
    # 1. Calculate Total IOPS (Max Read + Max Write is a rough worst-case estimate)
    # In reality, peaks might not align, but this is safe conservative sizing.
    max_read_iops = max([x['Maximum'] for x in read_iops]) if read_iops else 0
    max_write_iops = max([x['Maximum'] for x in write_iops]) if write_iops else 0
    total_max_iops = max_read_iops + max_write_iops
    
    storage_type = db_info.get('StorageType', 'aurora')
    allocated_storage = db_info.get('AllocatedStorage', 0)
    provisioned_iops = db_info.get('Iops', 0) # For io1, io2, gp3
    
    # --- IOPS Assessment ---
    print(f"Detected Storage: {storage_type.upper()} | Provisioned IOPS: {provisioned_iops if provisioned_iops else 'Baseline'}")
    
    if storage_type == 'gp2':
        # gp2 baseline is 3 IOPS per GB, min 100, max 16,000 (burstable to 3000)
        baseline_iops = max(100, min(allocated_storage * 3, 16000))
        if total_max_iops > baseline_iops:
            print(f"⚠️  [CRITICAL] IOPS Bound: Max IOPS ({total_max_iops:.0f}) exceeds gp2 baseline ({baseline_iops}). Latency likely.")
            print(f"   >>> Recommendation: Switch to gp3 or Provisioned IOPS (io1).")
        else:
            print(f"✅ IOPS within gp2 baseline limits.")
            
    elif storage_type in ['io1', 'io2', 'gp3'] and provisioned_iops:
        utilization = (total_max_iops / provisioned_iops) * 100
        if utilization > 90:
            print(f"⚠️  [CRITICAL] IOPS Saturation: Reaching {utilization:.1f}% of provisioned limit.")
            print(f"   >>> Recommendation: Increase Provisioned IOPS.")
        elif utilization < 40:
            print(f"💰 [SAVINGS] Over-provisioned IOPS. Peak usage is only {utilization:.1f}%.")
            print(f"   >>> Recommendation: Decrease Provisioned IOPS to save costs.")
        else:
            print(f"✅ IOPS sizing is healthy.")

    # --- Throughput Assessment ---
    # Simplified check for general limits (approx 500-1000 MB/s for many instances, varies by class)
    max_total_throughput_mb = (max([x['Maximum'] for x in read_thru], default=0) + \
                               max([x['Maximum'] for x in write_thru], default=0)) / 1024 / 1024
    
    print(f"Peak Throughput: {max_total_throughput_mb:.2f} MB/s")
    
    # --- Latency/Queue Assessment ---
    # Queue Depth > 5 usually implies disk cannot keep up with requests
    max_qd, avg_qd = 0, 0
    if queue_depth:
        vals = [x['Average'] for x in queue_depth]
        max_qd = max(vals)
        avg_qd = mean(vals)
    
    if avg_qd > 5:
        print(f"⚠️  [CRITICAL] High Queue Depth (Avg: {avg_qd:.2f}). Disk subsystem is a bottleneck.")
    elif max_qd > 20:
        print(f"⚠️  [WARNING] Spiky Queue Depth (Max: {max_qd:.2f}). Check for bursty reporting jobs.")
    else:
        print(f"✅ Queue Depth is healthy (Avg < 5).")

def main():
    args = get_args()
    
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    rds = session.client('rds')
    cw = session.client('cloudwatch')
    
    # 1. Get Instance Details
    print(f"Fetching details for: {args.db_identifier}...")
    try:
        db_resp = rds.describe_db_instances(DBInstanceIdentifier=args.db_identifier)
        db_info = db_resp['DBInstances'][0]
    except Exception as e:
        print(f"Error finding DB: {e}")
        sys.exit(1)

    # 2. Time Window (Last 15 days)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=15)
    
    print(f"Analyzing Metrics: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    print("-" * 60)

    # 3. Fetch Metrics
    # Note: Aurora uses 'VolumeReadIOPS' usually at cluster level, but 'ReadIOPS' at instance level works for standard RDS.
    # We assume Standard RDS metric names here.
    namespace = 'AWS/RDS'
    
    r_iops = get_cw_metric(cw, namespace, 'ReadIOPS', args.db_identifier, start_time, end_time)
    w_iops = get_cw_metric(cw, namespace, 'WriteIOPS', args.db_identifier, start_time, end_time)
    r_thru = get_cw_metric(cw, namespace, 'ReadThroughput', args.db_identifier, start_time, end_time)
    w_thru = get_cw_metric(cw, namespace, 'WriteThroughput', args.db_identifier, start_time, end_time)
    q_depth = get_cw_metric(cw, namespace, 'DiskQueueDepth', args.db_identifier, start_time, end_time)

    # 4. Visualize
    print_spark("Read IOPS", r_iops)
    print_spark("Write IOPS", w_iops)
    print_spark("Read Throughput", r_thru, unit='B/s')
    print_spark("Write Throughput", w_thru, unit='B/s')
    print_spark("Queue Depth", q_depth, key='Average', unit='Count') # QD is usually best viewed as Average
    
    # 5. Assess
    assess_io(db_info, r_iops, w_iops, r_thru, w_thru, q_depth)

if __name__ == "__main__":
    main()
=======

rds_cpu_analyzer.py
import boto3
import argparse
import sys
from datetime import datetime, timedelta
from statistics import mean
import sparklines  # pip install sparklines

def get_args():
    parser = argparse.ArgumentParser(description="RDS/Aurora CPU Rightsizing Analyzer")
    parser.add_argument("--profile", required=True, help="AWS SSO Profile Name")
    parser.add_argument("--region", required=True, help="AWS Region")
    parser.add_argument("--db-identifier", required=True, help="RDS Instance or Aurora Cluster Identifier")
    return parser.parse_args()

def get_db_config(session, db_id):
    """Fetches DB details and maps to EC2 spec to find vCPU/RAM"""
    rds = session.client('rds')
    ec2 = session.client('ec2')
    
    try:
        # 1. Get RDS details
        resp = rds.describe_db_instances(DBInstanceIdentifier=db_id)
        db = resp['DBInstances'][0]
        
        class_name = db['DBInstanceClass']
        engine = db['Engine']
        version = db['EngineVersion']
        
        # 2. Map to EC2 to get vCPU/RAM (Remove 'db.' prefix, e.g., db.m5.large -> m5.large)
        ec2_type = class_name.replace('db.', '')
        
        vcpu = "Unknown"
        ram = "Unknown"
        
        try:
            # Skip for 'serverless' classes if encountered
            if 'serverless' not in class_name:
                ec2_resp = ec2.describe_instance_types(InstanceTypes=[ec2_type])
                if ec2_resp['InstanceTypes']:
                    spec = ec2_resp['InstanceTypes'][0]
                    vcpu = spec['VCpuInfo']['DefaultVCpus']
                    ram_gb = spec['MemoryInfo']['SizeInMiB'] / 1024
                    ram = f"{ram_gb:.0f} GB"
        except Exception:
            pass # Fallback if EC2 lookup fails (e.g. some older/custom legacy classes)

        return {
            'id': db_id,
            'class': class_name,
            'engine': f"{engine} {version}",
            'vcpu': vcpu,
            'ram': ram,
            'is_burstable': class_name.startswith('db.t')
        }
    except Exception as e:
        print(f"Error fetching DB Config: {e}")
        sys.exit(1)

def get_cw_metric(cw_client, namespace, metric, db_id, start, end):
    try:
        response = cw_client.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric,
            Dimensions=[{'Name': 'DBInstanceIdentifier', 'Value': db_id}],
            StartTime=start,
            EndTime=end,
            Period=3600, # 1 Hour granularity
            Statistics=['Average', 'Maximum']
        )
        return sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
    except Exception:
        return []

def print_spark(label, data_points, key='Maximum', unit='%'):
    if not data_points:
        print(f"{label:<25} | No Data")
        return 0, 0

    values = [dp[key] for dp in data_points]
    avg_val = mean(values)
    max_val = max(values)
    
    spark = sparklines.sparklines(values)[0]
    print(f"{label:<25} |{spark}| Max: {max_val:.2f}{unit}, Avg: {avg_val:.2f}{unit}")
    return max_val, avg_val

def assess_cpu(config, max_cpu, avg_cpu, min_credits=None):
    print("\n--- 🧠 Rightsizing Assessment ---")
    
    # 1. Check Utilization
    if max_cpu < 40 and avg_cpu < 20:
        print(f"📉 [OVER-PROVISIONED] Instance is underutilized.")
        print(f"   Criteria: Peak CPU ({max_cpu:.1f}%) < 40% and Avg ({avg_cpu:.1f}%) < 20%.")
        print(f"   Recommendation: Downsize instance (e.g., to next smaller size in {config['class'].split('.')[1]} family).")
    
    elif avg_cpu > 70 or max_cpu > 95:
        print(f"📈 [UNDER-PROVISIONED] Instance is heavily loaded.")
        print(f"   Criteria: Avg CPU > 70% or Frequent spikes > 95%.")
        print(f"   Recommendation: Upsize instance or switch to Compute Optimized (c6g/c7g).")
    
    else:
        print(f"✅ [OPTIMIZED] CPU usage is within healthy bounds.")

    # 2. Check Burstable Credits (T-series only)
    if config['is_burstable']:
        print(f"\n--- Burstable (T-Series) Check ---")
        if min_credits is not None and min_credits < 50:
            print(f"⚠️  [CRITICAL] CPU Credits Exhausted (Min: {min_credits:.2f}).")
            print(f"   Performance is being throttled to baseline.")
            print(f"   Recommendation: Switch to Unlimited Mode or move to M/R instance family.")
        else:
            print(f"✅ Credit balance is healthy.")

def main():
    args = get_args()
    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    cw = session.client('cloudwatch')
    
    # 1. Get Configuration
    print("Fetching Configuration...")
    config = get_db_config(session, args.db_identifier)
    
    print("\n" + "="*60)
    print(f"DB Identifier  : {config['id']}")
    print(f"Engine         : {config['engine']}")
    print(f"Instance Class : {config['class']}")
    print(f"Hardware Spec  : {config['vcpu']} vCPU | {config['ram']} RAM")
    print("="*60 + "\n")

    # 2. Time Window
    end = datetime.utcnow()
    start = end - timedelta(days=15)
    
    # 3. Fetch Metrics
    cpu_data = get_cw_metric(cw, 'AWS/RDS', 'CPUUtilization', args.db_identifier, start, end)
    
    # Fetch credits only if T-series
    credit_data = []
    if config['is_burstable']:
        credit_data = get_cw_metric(cw, 'AWS/RDS', 'CPUCreditBalance', args.db_identifier, start, end)

    # 4. Visualize
    max_cpu, avg_cpu = print_spark("CPU Utilization", cpu_data, key='Maximum', unit='%')
    
    min_credits = None
    if config['is_burstable'] and credit_data:
        # For credits, we care about the Minimum balance
        vals = [x['Average'] for x in credit_data]
        min_credits = min(vals)
        spark = sparklines.sparklines(vals)[0]
        print(f"{'CPU Credit Balance':<25} |{spark}| Min: {min_credits:.2f}")

    # 5. Make Decisions
    assess_cpu(config, max_cpu, avg_cpu, min_credits)

if __name__ == "__main__":
    main()

========

python rds_cpu_analyzer.py --profile my-sso-profile --region us-east-1 --db-identifier my-db-instance




#!/usr/bin/env python3
'''
Merge several expression matrices into one matrix.
'''
import argparse
import pandas as pd
import numpy as np
import sys


def split_filename(filename):
    tokens = filename.split('.')
    return '.'.join(tokens[:-1]), tokens[-1]


def load_dataframe(filename):
    basename, ext = split_filename(filename)

    if ext == 'txt':
        # load dataframe from plaintext file
        return pd.read_csv(filename, index_col=0, sep='\t')
    elif ext == 'npy':
        # load data matrix from binary file
        X = np.load(filename)

        # load row names and column names from text files
        rownames = np.loadtxt('%s.rownames.txt' % basename, dtype=str)
        colnames = np.loadtxt('%s.colnames.txt' % basename, dtype=str)

        # combine data, row names, and column names into dataframe
        return pd.DataFrame(X, index=rownames, columns=colnames)
    else:
        print('error: filename %s is invalid' % (filename))
        sys.exit(-1)



def save_dataframe(filename, df):
    basename, ext = split_filename(filename)

    if ext == 'txt':
        # save dataframe to plaintext file
        df.to_csv(filename, sep='\t', na_rep='NA', float_format='%.8f')
    elif ext == 'npy':
    # save data matrix to binary file
        np.save(filename, df.to_numpy(dtype=np.float32))

        # save row names and column names to text files
        np.savetxt('%s.rownames.txt' % basename, df.index, fmt='%s')
        np.savetxt('%s.colnames.txt' % basename, df.columns, fmt='%s')
    else:
        print('error: filename %s is invalid' % (filename))
        sys.exit(-1)



def load_labels(filename):
    labels = pd.read_csv(filename, sep='\t', header=None, index_col=0)
    labels = labels[1].to_numpy()
    return labels

if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', help='input dataframes (genes x samples)', nargs='*')
    parser.add_argument('outfile', help='output dataframe')

    args = parser.parse_args()

    # initialize output expression matrix and labels
    X = pd.DataFrame()
    y = pd.DataFrame()

    # load each input file into expression matrix
    for infile in args.infiles:
        # load input file
        print('loading \'%s\'' % infile)

        X_i = pd.read_csv(infile, sep='\t', index_col=0)

        # remove extraneous columns
        X_i.drop(columns=['Entrez_Gene_Id'], inplace=True)

        # extract labels
        label = infile.split('.')[0].split('/')[-1]
        y_i = pd.DataFrame({'sample': X_i.columns, 'label': label})

        # append input dataframe to output dataframe
        X = pd.merge(X, X_i, left_index=True, right_index=True, how='outer')

        # append input labels to ouput labels
        y = pd.concat([y, y_i], ignore_index=True)

    y.set_index('sample', inplace=True)

    # save output expression matrix
    print('saving \'%s\'' % args.outfile)

    save_dataframe(args.outfile, X)
    save_dataframe('%s.labels.txt' % args.outfile.split('.')[0], y)
